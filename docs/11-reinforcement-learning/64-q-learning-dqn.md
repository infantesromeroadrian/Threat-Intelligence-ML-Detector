# Q-Learning y Deep Q-Networks (DQN)

## 1. Q-Learning: El Algoritmo Clasico

### Intuicion

```
+------------------------------------------------------------------------+
|  Q-LEARNING: Aprender el valor de cada accion                          |
+------------------------------------------------------------------------+
|                                                                        |
|  Objetivo: Aprender Q*(s, a) - el valor optimo de tomar accion 'a'     |
|            en estado 's' y luego actuar optimamente.                   |
|                                                                        |
|                                                                        |
|  Q-Table (para espacios pequenos):                                     |
|  ---------------------------------                                     |
|                                                                        |
|           |  Accion 0  |  Accion 1  |  Accion 2  |  Accion 3  |        |
|  ---------+------------+------------+------------+------------+        |
|  Estado 0 |    2.5     |    1.2     |    0.8     |    3.1     |        |
|  Estado 1 |    1.8     |    4.2     |    2.1     |    0.5     |        |
|  Estado 2 |    0.3     |    0.9     |    5.5     |    1.7     |        |
|  Estado 3 |    3.2     |    2.8     |    1.4     |    4.8     |        |
|  ---------+------------+------------+------------+------------+        |
|                                                                        |
|  Politica: pi(s) = argmax_a Q(s, a)                                    |
|                                                                        |
|  En estado 0: tomar accion 3 (Q=3.1 es el maximo)                      |
|  En estado 2: tomar accion 2 (Q=5.5 es el maximo)                      |
|                                                                        |
+------------------------------------------------------------------------+
```

### Ecuacion de Bellman para Q*

```
+------------------------------------------------------------------------+
|  ECUACION DE BELLMAN PARA Q*                                           |
+------------------------------------------------------------------------+
|                                                                        |
|  Q*(s, a) = E[r + gamma * max_a' Q*(s', a') | s, a]                    |
|                                                                        |
|  Donde:                                                                |
|    - r: recompensa inmediata                                           |
|    - gamma: factor de descuento                                        |
|    - s': estado siguiente                                              |
|    - max_a' Q*(s', a'): mejor valor futuro posible                     |
|                                                                        |
|                                                                        |
|  REGLA DE ACTUALIZACION Q-LEARNING:                                    |
|  ----------------------------------                                    |
|                                                                        |
|  Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)] |
|                                |__________________________________|    |
|                                         TD Error (delta)               |
|                                                                        |
|  alpha = learning rate                                                 |
|  TD Error = diferencia entre valor esperado y valor actual             |
|                                                                        |
|                                                                        |
|  INTUICION:                                                            |
|  ----------                                                            |
|                                                                        |
|  Valor actual    Estimacion nueva                                      |
|      |                 |                                               |
|      v                 v                                               |
|  Q(s,a) ---------> r + gamma * max Q(s',a')                            |
|                             ^                                          |
|                             |                                          |
|                   Recompensa + mejor futuro                            |
|                                                                        |
|  Mover Q(s,a) un poco hacia la estimacion nueva                        |
|                                                                        |
+------------------------------------------------------------------------+
```

### Algoritmo Q-Learning

```
+------------------------------------------------------------------------+
|  ALGORITMO Q-LEARNING TABULAR                                          |
+------------------------------------------------------------------------+
|                                                                        |
|  Inicializar Q(s, a) arbitrariamente (ej: zeros)                       |
|                                                                        |
|  Para cada episodio:                                                   |
|      Inicializar estado s                                              |
|                                                                        |
|      Mientras s no sea terminal:                                       |
|          |                                                             |
|          |  1. Elegir accion a usando politica derivada de Q           |
|          |     (ej: epsilon-greedy)                                    |
|          |                                                             |
|          |  2. Tomar accion a, observar r, s'                          |
|          |                                                             |
|          |  3. Actualizar Q:                                           |
|          |     Q(s,a) <- Q(s,a) + alpha * [r + gamma*max_a'Q(s',a') - Q(s,a)]
|          |                                                             |
|          |  4. s <- s'                                                 |
|          |                                                             |
|                                                                        |
|  Caracteristicas:                                                      |
|    - OFF-POLICY: usa max (greedy) para actualizar, no la accion real   |
|    - Model-free: no necesita conocer P(s'|s,a)                         |
|    - Convergencia garantizada (bajo ciertas condiciones)               |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion Q-Learning Tabular

```python
"""
Q-Learning Tabular completo con Gymnasium.
"""
import numpy as np
import gymnasium as gym
from typing import Tuple, List
import matplotlib.pyplot as plt
from collections import defaultdict


class QLearningAgent:
    """
    Agente Q-Learning tabular.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Args:
            n_actions: Numero de acciones posibles
            alpha: Learning rate
            gamma: Factor de descuento
            epsilon_*: Parametros para epsilon-greedy
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table como diccionario (soporta estados continuos discretizados)
        self.q_table: dict = defaultdict(lambda: np.zeros(n_actions))

    def discretize_state(self, state: np.ndarray, bins: List[np.ndarray]) -> Tuple:
        """
        Discretiza estado continuo.

        Args:
            state: Estado continuo
            bins: Bins para cada dimension

        Returns:
            Estado discretizado como tupla
        """
        discrete = []
        for i, val in enumerate(state):
            discrete.append(np.digitize(val, bins[i]))
        return tuple(discrete)

    def select_action(self, state: Tuple) -> int:
        """
        Selecciona accion usando epsilon-greedy.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool
    ) -> float:
        """
        Actualiza Q-table.

        Returns:
            TD error
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD Error
        td_error = target - current_q

        # Update
        self.q_table[state][action] = current_q + self.alpha * td_error

        return td_error

    def decay_epsilon(self) -> None:
        """Reduce epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_q_learning(
    env_name: str = "CartPole-v1",
    n_episodes: int = 1000,
    max_steps: int = 500
) -> Tuple[QLearningAgent, List[float]]:
    """
    Entrena agente Q-Learning.

    Returns:
        (agente_entrenado, historial_recompensas)
    """
    env = gym.make(env_name)

    # Crear bins para discretizacion (CartPole tiene 4 dimensiones)
    # [posicion_carro, velocidad_carro, angulo_palo, velocidad_angular]
    n_bins = 20
    bins = [
        np.linspace(-4.8, 4.8, n_bins),      # posicion
        np.linspace(-4, 4, n_bins),           # velocidad
        np.linspace(-0.418, 0.418, n_bins),   # angulo
        np.linspace(-4, 4, n_bins)            # velocidad angular
    ]

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    rewards_history = []
    td_errors = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        state = agent.discretize_state(state, bins)

        total_reward = 0
        episode_td_errors = []

        for step in range(max_steps):
            # Seleccionar accion
            action = agent.select_action(state)

            # Ejecutar accion
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = agent.discretize_state(next_state, bins)

            # Modificar reward para mejor aprendizaje
            if done and step < max_steps - 1:
                reward = -100  # Penalizar caida

            # Actualizar Q-table
            td_error = agent.update(state, action, reward, next_state, done)
            episode_td_errors.append(abs(td_error))

            total_reward += reward
            state = next_state

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        rewards_history.append(total_reward)
        td_errors.append(np.mean(episode_td_errors))

        # Log cada 100 episodios
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Q-states: {len(agent.q_table)}")

    env.close()
    return agent, rewards_history


def plot_training(rewards: List[float], window: int = 100) -> None:
    """Grafica curvas de aprendizaje."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rewards por episodio
    axes[0].plot(rewards, alpha=0.3, label='Reward por episodio')

    # Media movil
    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards,
            np.ones(window) / window,
            mode='valid'
        )
        axes[0].plot(
            range(window - 1, len(rewards)),
            moving_avg,
            label=f'Media movil ({window} eps)',
            color='red',
            linewidth=2
        )

    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Curva de Aprendizaje Q-Learning')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histograma de rewards finales
    axes[1].hist(rewards[-100:], bins=20, edgecolor='black')
    axes[1].axvline(
        np.mean(rewards[-100:]),
        color='red',
        linestyle='--',
        label=f'Media: {np.mean(rewards[-100:]):.1f}'
    )
    axes[1].set_xlabel('Recompensa')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribucion de Rewards (ultimos 100 eps)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('q_learning_training.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=== Entrenando Q-Learning en CartPole ===\n")
    agent, rewards = train_q_learning(n_episodes=1000)

    print(f"\nResultados finales:")
    print(f"  Media ultimos 100 episodios: {np.mean(rewards[-100:]):.2f}")
    print(f"  Estados visitados: {len(agent.q_table)}")

    plot_training(rewards)
```

## 2. SARSA: La Alternativa On-Policy

### Diferencia con Q-Learning

```
+------------------------------------------------------------------------+
|  Q-LEARNING vs SARSA                                                   |
+------------------------------------------------------------------------+
|                                                                        |
|  Q-LEARNING (Off-Policy):                                              |
|  -------------------------                                             |
|  Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]     |
|                                       ^^^                              |
|                              Usa MEJOR accion posible                  |
|                              (no la que realmente tomara)              |
|                                                                        |
|                                                                        |
|  SARSA (On-Policy):                                                    |
|  ------------------                                                    |
|  Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]            |
|                                           ^^                           |
|                              Usa accion que REALMENTE tomara           |
|                              (State-Action-Reward-State-Action)        |
|                                                                        |
|                                                                        |
|  IMPLICACIONES:                                                        |
|  --------------                                                        |
|                                                                        |
|  Q-Learning:                          SARSA:                           |
|  - Aprende politica optima            - Aprende politica actual        |
|  - Mas agresivo                       - Mas conservador                |
|  - Puede aprender de datos            - Solo de su propia              |
|    de cualquier politica                experiencia                    |
|  - Puede ser inestable                - Mas estable                    |
|                                                                        |
|                                                                        |
|  Ejemplo - Cliff Walking:                                              |
|  +---+---+---+---+---+---+                                             |
|  | S |   |   |   |   | G |   Q-Learning: va por el borde (optimo      |
|  +---+---+---+---+---+---+              pero peligroso con epsilon)    |
|  |   | C | C | C | C |   |                                             |
|  +---+---+---+---+---+---+   SARSA: va por arriba (mas seguro          |
|                                     considerando exploracion)          |
|  S=Start, G=Goal, C=Cliff                                              |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion SARSA

```python
"""
SARSA - On-Policy TD Control.
"""
import numpy as np
import gymnasium as gym
from typing import Tuple, List
from collections import defaultdict


class SARSAAgent:
    """
    Agente SARSA.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state: Tuple) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        next_action: int,
        done: bool
    ) -> float:
        """
        SARSA update.

        Diferencia clave: usa next_action (la accion que realmente tomara)
        en lugar de max Q.
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # SARSA: usa Q(s', a') no max_a' Q(s', a')
            target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

        return td_error


def train_sarsa_vs_qlearning(env_name: str = "CliffWalking-v0", n_episodes: int = 500):
    """Compara SARSA vs Q-Learning en Cliff Walking."""
    env = gym.make(env_name)

    # Inicializar agentes
    sarsa_agent = SARSAAgent(n_actions=env.action_space.n, epsilon=0.1)
    qlearning_agent = QLearningAgent(
        n_actions=env.action_space.n,
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=1.0  # Sin decay para comparacion justa
    )

    sarsa_rewards = []
    qlearning_rewards = []

    for episode in range(n_episodes):
        # SARSA
        state, _ = env.reset()
        state = (state,)
        action = sarsa_agent.select_action(state)
        sarsa_total = 0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = (next_state,)
            next_action = sarsa_agent.select_action(next_state)

            sarsa_agent.update(state, action, reward, next_state, next_action, done)

            sarsa_total += reward
            state = next_state
            action = next_action

            if done:
                break

        sarsa_rewards.append(sarsa_total)

        # Q-Learning
        state, _ = env.reset()
        state = (state,)
        qlearning_total = 0

        while True:
            action = qlearning_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = (next_state,)

            qlearning_agent.update(state, action, reward, next_state, done)

            qlearning_total += reward
            state = next_state

            if done:
                break

        qlearning_rewards.append(qlearning_total)

    env.close()

    # Plot comparacion
    window = 50
    sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    q_smooth = np.convolve(qlearning_rewards, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_smooth, label='SARSA (On-Policy)', linewidth=2)
    plt.plot(q_smooth, label='Q-Learning (Off-Policy)', linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa (media movil)')
    plt.title('SARSA vs Q-Learning en Cliff Walking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sarsa_vs_qlearning.png', dpi=150)
    plt.show()

    return sarsa_rewards, qlearning_rewards
```

## 3. Deep Q-Network (DQN)

### Motivacion: Limitaciones de Q-Learning Tabular

```
+------------------------------------------------------------------------+
|  POR QUE NECESITAMOS DQN?                                              |
+------------------------------------------------------------------------+
|                                                                        |
|  PROBLEMA: Q-Learning tabular no escala                                |
|                                                                        |
|  Ejemplo - Atari (imagen 84x84 en escala de grises):                   |
|    Estados posibles = 256^(84*84) = 256^7056                           |
|    = mas atomos que en el universo                                     |
|                                                                        |
|  Solucion: Aproximar Q(s,a) con una RED NEURONAL                       |
|                                                                        |
|                                                                        |
|  Q-LEARNING TABULAR:           DQN:                                    |
|  -------------------           ----                                    |
|                                                                        |
|  +-------+-------+             +---------------+                       |
|  |Estado |Q-value|             |               |                       |
|  +-------+-------+             |  Red Neuronal |                       |
|  |  s1   |  2.5  |    vs       |               |                       |
|  |  s2   |  1.8  |             |  s --> Q(s,a) |                       |
|  | ...   | ...   |             |  para todo a  |                       |
|  +-------+-------+             |               |                       |
|                                +---------------+                       |
|  No escala                     Generaliza a estados no vistos          |
|                                                                        |
+------------------------------------------------------------------------+
```

### Arquitectura DQN

```
+------------------------------------------------------------------------+
|  ARQUITECTURA DQN                                                      |
+------------------------------------------------------------------------+
|                                                                        |
|  Estado (imagen o vector)                                              |
|         |                                                              |
|         v                                                              |
|  +------------------+                                                  |
|  |   Conv2D o       |  (Para imagenes: capas convolucionales)          |
|  |   Dense          |  (Para vectores: capas densas)                   |
|  +------------------+                                                  |
|         |                                                              |
|         v                                                              |
|  +------------------+                                                  |
|  |   Hidden Layers  |  (ReLU activation)                               |
|  |   Dense(512)     |                                                  |
|  |   Dense(256)     |                                                  |
|  +------------------+                                                  |
|         |                                                              |
|         v                                                              |
|  +------------------+                                                  |
|  |   Output Layer   |  (Sin activacion - valores crudos)               |
|  |   Dense(n_actions)|                                                 |
|  +------------------+                                                  |
|         |                                                              |
|         v                                                              |
|  [Q(s,a1), Q(s,a2), ..., Q(s,an)]                                      |
|                                                                        |
|  Un Q-value por cada accion posible                                    |
|                                                                        |
+------------------------------------------------------------------------+
```

### Innovaciones Clave de DQN

```
+------------------------------------------------------------------------+
|  INNOVACIONES DE DQN (DeepMind, 2015)                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  1. EXPERIENCE REPLAY                                                  |
|     ------------------                                                 |
|     Guardar transiciones (s, a, r, s', done) en buffer                 |
|     Muestrear mini-batches aleatorios para entrenar                    |
|                                                                        |
|     +--------------------------------------------+                     |
|     |  Replay Buffer (capacidad = 100,000)       |                     |
|     +--------------------------------------------+                     |
|     | (s1, a1, r1, s1', done1)                   |                     |
|     | (s2, a2, r2, s2', done2)                   |                     |
|     | ...                                         |                     |
|     | (sN, aN, rN, sN', doneN)                   |                     |
|     +--------------------------------------------+                     |
|              |                                                         |
|              v                                                         |
|     [Sample batch de 32 aleatorios]                                    |
|                                                                        |
|     Beneficios:                                                        |
|       - Rompe correlacion temporal de datos                            |
|       - Reutiliza experiencias (eficiente)                             |
|       - Estabiliza entrenamiento                                       |
|                                                                        |
|                                                                        |
|  2. TARGET NETWORK                                                     |
|     ---------------                                                    |
|     Dos redes: Q (online) y Q' (target)                                |
|     Q' se actualiza lentamente (copia cada N pasos)                    |
|                                                                        |
|     +-----------+                    +-----------+                     |
|     |  Q-Network|                    |Q'-Network |                     |
|     |  (online) |                    | (target)  |                     |
|     +-----------+                    +-----------+                     |
|          |                                |                            |
|          v                                v                            |
|     Prediccion Q(s,a)              Target: r + gamma*max Q'(s',a')     |
|                                                                        |
|     Loss = (Q(s,a) - target)^2                                         |
|                                                                        |
|     Cada C pasos: Q' <- Q (copia de pesos)                             |
|                                                                        |
|     Beneficios:                                                        |
|       - Targets mas estables                                           |
|       - Evita "perseguir un objetivo movil"                            |
|       - Mejora convergencia                                            |
|                                                                        |
+------------------------------------------------------------------------+
```

### Algoritmo DQN Completo

```
+------------------------------------------------------------------------+
|  ALGORITMO DQN                                                         |
+------------------------------------------------------------------------+
|                                                                        |
|  Inicializar:                                                          |
|    - Q-network con pesos theta                                         |
|    - Target network Q' con pesos theta' = theta                        |
|    - Replay buffer D                                                   |
|                                                                        |
|  Para cada episodio:                                                   |
|      Inicializar estado s                                              |
|                                                                        |
|      Para cada paso t:                                                 |
|          |                                                             |
|          |  1. Seleccionar accion a:                                   |
|          |     - Con prob epsilon: accion aleatoria                    |
|          |     - Con prob 1-epsilon: argmax_a Q(s, a; theta)           |
|          |                                                             |
|          |  2. Ejecutar a, observar r, s'                              |
|          |                                                             |
|          |  3. Guardar transicion (s, a, r, s', done) en D             |
|          |                                                             |
|          |  4. Muestrear mini-batch de D                               |
|          |                                                             |
|          |  5. Calcular targets:                                       |
|          |     y_j = r_j                    si done                    |
|          |     y_j = r_j + gamma*max_a' Q'(s'_j, a'; theta')  si no   |
|          |                                                             |
|          |  6. Hacer paso de gradiente en:                             |
|          |     Loss = (1/N) * sum_j (y_j - Q(s_j, a_j; theta))^2       |
|          |                                                             |
|          |  7. Cada C pasos: theta' <- theta                           |
|          |                                                             |
|          |  8. Decay epsilon                                           |
|          |                                                             |
|          |  s <- s'                                                    |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion DQN con PyTorch

```python
"""
DQN completo con PyTorch y Gymnasium.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
from typing import Tuple, List, Deque
import matplotlib.pyplot as plt


class DQN(nn.Module):
    """
    Red neuronal para aproximar Q(s, a).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: estado -> Q-values para todas las acciones."""
        return self.network(state)


class ReplayBuffer:
    """
    Experience Replay Buffer.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Guarda transicion en buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Muestrea batch aleatorio."""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN completo.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "auto"
    ):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon para exploracion
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Redes
        self.q_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Contador para target update
        self.steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selecciona accion con epsilon-greedy.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Guarda transicion en replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """
        Un paso de entrenamiento.

        Returns:
            Loss si se entreno, None si no hay suficientes samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-values actuales: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q'(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss (MSE)
        loss = nn.MSELoss()(current_q, target_q)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (estabilidad)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        """Reduce epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_dqn(
    env_name: str = "CartPole-v1",
    n_episodes: int = 500,
    max_steps: int = 500,
    render_freq: int = 0
) -> Tuple[DQNAgent, List[float]]:
    """
    Entrena agente DQN.
    """
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100
    )

    rewards_history = []
    losses = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []

        for step in range(max_steps):
            # Seleccionar y ejecutar accion
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Guardar transicion
            agent.store_transition(state, action, reward, next_state, done)

            # Entrenar
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        rewards_history.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))

        # Log
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_loss = np.mean(losses[-50:]) if losses else 0
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, rewards_history


def evaluate_agent(agent: DQNAgent, env_name: str, n_episodes: int = 10) -> float:
    """Evalua agente entrenado."""
    env = gym.make(env_name, render_mode="human")

    total_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Evaluation episode {episode + 1}: Reward = {episode_reward}")

    env.close()
    return np.mean(total_rewards)


if __name__ == "__main__":
    print("=== Entrenando DQN en CartPole ===\n")
    agent, rewards = train_dqn(n_episodes=500)

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('DQN - Curva de Aprendizaje')
    plt.grid(True, alpha=0.3)
    plt.savefig('dqn_training.png', dpi=150)
    plt.show()

    # Evaluar
    print("\n=== Evaluando agente ===")
    avg_reward = evaluate_agent(agent, "CartPole-v1", n_episodes=5)
    print(f"\nRecompensa media en evaluacion: {avg_reward:.2f}")
```

## 4. Mejoras a DQN

### Double DQN

```
+------------------------------------------------------------------------+
|  DOUBLE DQN                                                            |
+------------------------------------------------------------------------+
|                                                                        |
|  PROBLEMA con DQN:                                                     |
|  -----------------                                                     |
|  Target = r + gamma * max_a' Q(s', a'; theta')                         |
|                        ^^^                                             |
|                    Misma red selecciona y evalua                       |
|                    -> Sobreestimacion de Q-values                      |
|                                                                        |
|                                                                        |
|  SOLUCION Double DQN:                                                  |
|  --------------------                                                  |
|  Desacoplar seleccion de evaluacion                                    |
|                                                                        |
|  1. Online network (theta) SELECCIONA mejor accion:                    |
|     a* = argmax_a' Q(s', a'; theta)                                    |
|                                                                        |
|  2. Target network (theta') EVALUA esa accion:                         |
|     Target = r + gamma * Q(s', a*; theta')                             |
|                                                                        |
|                                                                        |
|  COMPARACION:                                                          |
|  ------------                                                          |
|                                                                        |
|  DQN:        r + gamma * max_a' Q(s', a'; theta')                      |
|                          ^^^^^^^^^^^^^^^^^^^^^^^^                      |
|                          theta' selecciona y evalua                    |
|                                                                        |
|  Double DQN: r + gamma * Q(s', argmax_a' Q(s', a'; theta); theta')     |
|                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^            |
|                                theta selecciona, theta' evalua         |
|                                                                        |
+------------------------------------------------------------------------+
```

### Dueling DQN

```
+------------------------------------------------------------------------+
|  DUELING DQN                                                           |
+------------------------------------------------------------------------+
|                                                                        |
|  IDEA: Separar Q(s,a) en dos componentes                               |
|                                                                        |
|  Q(s, a) = V(s) + A(s, a)                                              |
|                                                                        |
|  Donde:                                                                |
|    V(s) = Value function - "que tan bueno es el estado"                |
|    A(s,a) = Advantage - "que tan buena es la accion vs promedio"       |
|                                                                        |
|                                                                        |
|  ARQUITECTURA:                                                         |
|  -------------                                                         |
|                                                                        |
|  Estado                                                                |
|    |                                                                   |
|    v                                                                   |
|  +----------------+                                                    |
|  | Shared Layers  |                                                    |
|  +----------------+                                                    |
|    |           |                                                       |
|    v           v                                                       |
|  +------+   +------+                                                   |
|  |Value |   |Advt. |                                                   |
|  |Stream|   |Stream|                                                   |
|  +------+   +------+                                                   |
|    |           |                                                       |
|    v           v                                                       |
|  V(s)      A(s,a1), A(s,a2), ...                                       |
|    |           |                                                       |
|    +-----------+                                                       |
|          |                                                             |
|          v                                                             |
|  Q(s,a) = V(s) + (A(s,a) - mean_a'(A(s,a')))                          |
|                   ^^^^^^^^^^^^^^^^^^^^^^^^^^                           |
|                   Centramos A para identificabilidad                   |
|                                                                        |
|                                                                        |
|  BENEFICIO:                                                            |
|  ----------                                                            |
|  - Muchos estados tienen valor similar independiente de la accion      |
|  - Aprende V(s) mas eficientemente                                     |
|  - Mejor generalizacion                                                |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion Double Dueling DQN

```python
"""
Double Dueling DQN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.

    Separa Value stream y Advantage stream.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Feature layer compartida
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        features = self.feature_layer(state)

        value = self.value_stream(features)  # (batch, 1)
        advantages = self.advantage_stream(features)  # (batch, n_actions)

        # Combinar: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class DoubleDuelingDQNAgent:
    """
    Agente con Double DQN + Dueling architecture.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Dueling networks
        self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def train_step(self) -> float | None:
        """
        Double DQN training step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: seleccionar accion con online, evaluar con target
        with torch.no_grad():
            # Online network selecciona mejor accion
            next_actions = self.q_network(next_states).argmax(dim=1)

            # Target network evalua esa accion
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    # ... resto de metodos igual que DQNAgent
```

## 5. Prioritized Experience Replay (PER)

```
+------------------------------------------------------------------------+
|  PRIORITIZED EXPERIENCE REPLAY                                         |
+------------------------------------------------------------------------+
|                                                                        |
|  IDEA: No todas las experiencias son igual de utiles                   |
|        Muestrear MAS las experiencias con alto TD error                |
|                                                                        |
|                                                                        |
|  PRIORIDAD basada en TD error:                                         |
|  -----------------------------                                         |
|                                                                        |
|  p_i = |delta_i| + epsilon                                             |
|                                                                        |
|  donde delta_i = |r + gamma*max Q'(s',a') - Q(s,a)|                    |
|                                                                        |
|                                                                        |
|  PROBABILIDAD de muestreo:                                             |
|  --------------------------                                            |
|                                                                        |
|  P(i) = p_i^alpha / sum_k p_k^alpha                                    |
|                                                                        |
|  alpha = 0: muestreo uniforme (replay normal)                          |
|  alpha = 1: muestreo proporcional a prioridad                          |
|                                                                        |
|                                                                        |
|  IMPORTANCE SAMPLING (IS) weights:                                     |
|  ---------------------------------                                     |
|  Corregir sesgo introducido por muestreo no uniforme                   |
|                                                                        |
|  w_i = (1/N * 1/P(i))^beta                                             |
|                                                                        |
|  Loss = w_i * (y_i - Q(s_i, a_i))^2                                    |
|                                                                        |
|  beta: 0 -> 1 (annealing durante entrenamiento)                        |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion PER

```python
"""
Prioritized Experience Replay con SumTree.
"""
import numpy as np
from typing import Tuple


class SumTree:
    """
    SumTree para muestreo eficiente por prioridad.

    Estructura de arbol donde cada nodo padre es la suma de sus hijos.
    Permite muestreo proporcional en O(log n).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Arbol binario
        self.data = np.zeros(capacity, dtype=object)  # Datos
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagar cambio hacia arriba."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Encontrar leaf node para valor s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        """Suma total de prioridades."""
        return self.tree[0]

    def add(self, priority: float, data: Tuple) -> None:
        """Agregar dato con prioridad."""
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Actualizar prioridad de un nodo."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Tuple]:
        """Obtener dato para valor s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Beta annealing."""
        return min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * self.frame / self.beta_frames
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Agregar transicion con prioridad maxima."""
        data = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size: int) -> Tuple:
        """
        Muestrear batch con prioridad.

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size

        self.frame += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calcular importance sampling weights
        probs = np.array(priorities) / self.tree.total
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalizar

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices: list, td_errors: np.ndarray) -> None:
        """Actualizar prioridades basado en TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
```

## 6. Aplicacion: Agente de Deteccion de Intrusiones

```python
"""
DQN para deteccion de intrusiones en red.

El agente aprende a clasificar trafico como normal o ataque,
optimizando el balance entre deteccion y falsas alarmas.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List
from dataclasses import dataclass
from enum import Enum


class IDSAction(Enum):
    """Acciones del IDS."""
    ALLOW = 0  # Permitir trafico
    ALERT = 1  # Alertar (posible ataque)
    BLOCK = 2  # Bloquear (ataque confirmado)


@dataclass
class NetworkFlow:
    """Representa un flujo de red."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    bytes_sent: int
    bytes_recv: int
    packets: int
    duration: float
    flags: int
    is_attack: bool  # Ground truth (para simulacion)


class IDSEnvironment:
    """
    Entorno simulado para IDS basado en RL.

    Estado: features del flujo de red
    Accion: ALLOW, ALERT, BLOCK
    Recompensa: basada en deteccion correcta vs falsas alarmas
    """

    def __init__(
        self,
        attack_ratio: float = 0.3,
        false_positive_cost: float = -5,
        false_negative_cost: float = -50,
        true_positive_reward: float = 10,
        true_negative_reward: float = 1
    ):
        self.attack_ratio = attack_ratio
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        self.tp_reward = true_positive_reward
        self.tn_reward = true_negative_reward

        self.state_dim = 10  # Features del flujo
        self.action_dim = 3

        self.current_flow: NetworkFlow | None = None
        self.episode_length = 100
        self.step_count = 0

    def _generate_flow(self) -> NetworkFlow:
        """Genera flujo de red (normal o ataque)."""
        is_attack = np.random.random() < self.attack_ratio

        if is_attack:
            # Patrones de ataque
            flow = NetworkFlow(
                src_ip=f"10.0.0.{np.random.randint(1, 255)}",
                dst_ip="192.168.1.1",
                src_port=np.random.randint(1024, 65535),
                dst_port=np.random.choice([22, 23, 80, 443, 3389]),  # Puertos comunes
                protocol=np.random.choice([6, 17]),  # TCP/UDP
                bytes_sent=np.random.randint(10000, 1000000),  # Alto volumen
                bytes_recv=np.random.randint(100, 1000),
                packets=np.random.randint(100, 10000),
                duration=np.random.uniform(0.1, 5.0),
                flags=np.random.randint(0, 63),
                is_attack=True
            )
        else:
            # Trafico normal
            flow = NetworkFlow(
                src_ip=f"192.168.1.{np.random.randint(2, 254)}",
                dst_ip=f"10.0.0.{np.random.randint(1, 255)}",
                src_port=np.random.randint(1024, 65535),
                dst_port=np.random.choice([80, 443, 8080]),
                protocol=6,  # TCP
                bytes_sent=np.random.randint(100, 10000),
                bytes_recv=np.random.randint(1000, 50000),
                packets=np.random.randint(5, 100),
                duration=np.random.uniform(1.0, 60.0),
                flags=2,  # SYN normal
                is_attack=False
            )

        return flow

    def _flow_to_state(self, flow: NetworkFlow) -> np.ndarray:
        """Convierte flujo a vector de estado normalizado."""
        state = np.array([
            hash(flow.src_ip) % 1000 / 1000,  # IP hash normalizado
            hash(flow.dst_ip) % 1000 / 1000,
            flow.src_port / 65535,
            flow.dst_port / 65535,
            flow.protocol / 255,
            np.log1p(flow.bytes_sent) / 20,
            np.log1p(flow.bytes_recv) / 20,
            np.log1p(flow.packets) / 10,
            flow.duration / 60,
            flow.flags / 63
        ], dtype=np.float32)

        return state

    def reset(self) -> np.ndarray:
        """Reinicia entorno."""
        self.step_count = 0
        self.current_flow = self._generate_flow()
        return self._flow_to_state(self.current_flow)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta accion del IDS.

        Returns:
            (nuevo_estado, recompensa, done, info)
        """
        action = IDSAction(action)
        is_attack = self.current_flow.is_attack

        # Calcular recompensa basada en matriz de confusion
        if action == IDSAction.ALLOW:
            if is_attack:
                reward = self.fn_cost  # False Negative (peligroso!)
                result = "FN"
            else:
                reward = self.tn_reward  # True Negative
                result = "TN"
        elif action == IDSAction.ALERT:
            if is_attack:
                reward = self.tp_reward * 0.5  # True Positive (alerta)
                result = "TP_ALERT"
            else:
                reward = self.fp_cost * 0.3  # False Positive (menos grave que block)
                result = "FP_ALERT"
        else:  # BLOCK
            if is_attack:
                reward = self.tp_reward  # True Positive (bloqueado!)
                result = "TP_BLOCK"
            else:
                reward = self.fp_cost  # False Positive (bloqueo incorrecto)
                result = "FP_BLOCK"

        self.step_count += 1
        done = self.step_count >= self.episode_length

        # Generar siguiente flujo
        self.current_flow = self._generate_flow()
        next_state = self._flow_to_state(self.current_flow)

        info = {
            "result": result,
            "is_attack": is_attack,
            "action": action.name
        }

        return next_state, reward, done, info


def train_ids_agent(n_episodes: int = 500) -> Tuple[DQNAgent, dict]:
    """Entrena agente IDS con DQN."""
    env = IDSEnvironment()

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )

    metrics = {
        "rewards": [],
        "tp": [],
        "tn": [],
        "fp": [],
        "fn": []
    }

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        episode_results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            total_reward += reward

            # Contar resultados
            result = info["result"]
            if "TP" in result:
                episode_results["TP"] += 1
            elif "TN" in result:
                episode_results["TN"] += 1
            elif "FP" in result:
                episode_results["FP"] += 1
            else:
                episode_results["FN"] += 1

            state = next_state
            if done:
                break

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        for key in episode_results:
            metrics[key.lower()].append(episode_results[key])

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(metrics["rewards"][-50:])
            precision = np.sum(metrics["tp"][-50:]) / (
                np.sum(metrics["tp"][-50:]) + np.sum(metrics["fp"][-50:]) + 1e-8
            )
            recall = np.sum(metrics["tp"][-50:]) / (
                np.sum(metrics["tp"][-50:]) + np.sum(metrics["fn"][-50:]) + 1e-8
            )

            print(f"Episode {episode + 1} | "
                  f"Reward: {avg_reward:.1f} | "
                  f"Precision: {precision:.2f} | "
                  f"Recall: {recall:.2f}")

    return agent, metrics


if __name__ == "__main__":
    print("=== Entrenando IDS con DQN ===\n")
    agent, metrics = train_ids_agent(n_episodes=300)

    # Graficar metricas
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rewards
    axes[0, 0].plot(metrics["rewards"], alpha=0.3)
    axes[0, 0].plot(np.convolve(metrics["rewards"], np.ones(20)/20, mode='valid'))
    axes[0, 0].set_title("Recompensa por Episodio")
    axes[0, 0].set_xlabel("Episodio")

    # Confusion matrix counts
    axes[0, 1].plot(np.convolve(metrics["tp"], np.ones(20)/20, mode='valid'), label="TP")
    axes[0, 1].plot(np.convolve(metrics["tn"], np.ones(20)/20, mode='valid'), label="TN")
    axes[0, 1].plot(np.convolve(metrics["fp"], np.ones(20)/20, mode='valid'), label="FP")
    axes[0, 1].plot(np.convolve(metrics["fn"], np.ones(20)/20, mode='valid'), label="FN")
    axes[0, 1].legend()
    axes[0, 1].set_title("Matriz de Confusion (Media Movil)")

    # Precision y Recall
    precision = [tp / (tp + fp + 1e-8) for tp, fp in zip(metrics["tp"], metrics["fp"])]
    recall = [tp / (tp + fn + 1e-8) for tp, fn in zip(metrics["tp"], metrics["fn"])]

    axes[1, 0].plot(np.convolve(precision, np.ones(20)/20, mode='valid'), label="Precision")
    axes[1, 0].plot(np.convolve(recall, np.ones(20)/20, mode='valid'), label="Recall")
    axes[1, 0].legend()
    axes[1, 0].set_title("Precision y Recall")

    # F1 Score
    f1 = [2*p*r/(p+r+1e-8) for p, r in zip(precision, recall)]
    axes[1, 1].plot(np.convolve(f1, np.ones(20)/20, mode='valid'))
    axes[1, 1].set_title("F1 Score")
    axes[1, 1].set_xlabel("Episodio")

    plt.tight_layout()
    plt.savefig("ids_dqn_training.png", dpi=150)
    plt.show()
```

## 7. Resumen

```
+------------------------------------------------------------------------+
|  Q-LEARNING Y DQN - RESUMEN                                            |
+------------------------------------------------------------------------+
|                                                                        |
|  Q-LEARNING TABULAR:                                                   |
|    - Almacena Q(s,a) en tabla                                          |
|    - Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma*maxQ(s',a') - Q(s,a)]|
|    - Off-policy: usa max para target                                   |
|    - Solo funciona para espacios pequenos                              |
|                                                                        |
|  SARSA:                                                                |
|    - On-policy: usa accion real a' para target                         |
|    - Mas conservador y estable                                         |
|    - Considera exploracion en el aprendizaje                           |
|                                                                        |
|  DQN:                                                                  |
|    - Red neuronal aproxima Q(s,a)                                      |
|    - Experience Replay: buffer de transiciones                         |
|    - Target Network: estabiliza entrenamiento                          |
|    - Escala a espacios continuos/grandes                               |
|                                                                        |
|  MEJORAS A DQN:                                                        |
|    - Double DQN: desacopla seleccion de evaluacion                     |
|    - Dueling DQN: separa V(s) y A(s,a)                                 |
|    - PER: muestrea experiencias importantes mas frecuentemente         |
|                                                                        |
|  APLICACIONES EN SEGURIDAD:                                            |
|    - IDS adaptativo                                                    |
|    - Network defense                                                   |
|    - Automated response                                                |
|                                                                        |
+------------------------------------------------------------------------+
```

---

**Siguiente:** Policy Gradient y PPO
