# Policy Gradient Methods

## 1. Introduccion a Policy Gradient

### Motivacion: Por que aprender la politica directamente?

```
+------------------------------------------------------------------------+
|  VALUE-BASED vs POLICY-BASED                                           |
+------------------------------------------------------------------------+
|                                                                        |
|  VALUE-BASED (Q-Learning, DQN):                                        |
|  ------------------------------                                        |
|  1. Aprende Q(s,a) para todas las acciones                             |
|  2. Deriva politica: pi(s) = argmax_a Q(s,a)                           |
|                                                                        |
|  LIMITACIONES:                                                         |
|    - Requiere enumerar TODAS las acciones (dificil si continuas)       |
|    - Politica siempre deterministica (argmax)                          |
|    - Pequenos cambios en Q pueden causar grandes cambios en politica   |
|                                                                        |
|                                                                        |
|  POLICY-BASED (Policy Gradient):                                       |
|  -------------------------------                                       |
|  1. Aprende pi(a|s; theta) DIRECTAMENTE                                |
|  2. Parametriza la politica con red neuronal                           |
|                                                                        |
|  VENTAJAS:                                                             |
|    - Funciona con acciones CONTINUAS                                   |
|    - Puede aprender politicas ESTOCASTICAS                             |
|    - Convergencia mas suave                                            |
|    - Mejor para algunos entornos (parcialmente observables)            |
|                                                                        |
|                                                                        |
|  EJEMPLO - Robot con brazo:                                            |
|  --------------------------                                            |
|                                                                        |
|  Value-based:   No puede (infinitas acciones de angulo/fuerza)         |
|                                                                        |
|  Policy-based:  pi(s) = Normal(mu(s), sigma(s))                        |
|                 Red neuronal predice mu y sigma                        |
|                 Accion = sample de la distribucion                     |
|                                                                        |
+------------------------------------------------------------------------+
```

### Politica Parametrizada

```
+------------------------------------------------------------------------+
|  POLITICA PARAMETRIZADA pi(a|s; theta)                                 |
+------------------------------------------------------------------------+
|                                                                        |
|  Para acciones DISCRETAS:                                              |
|  -------------------------                                             |
|                                                                        |
|  Estado s                                                              |
|      |                                                                 |
|      v                                                                 |
|  +----------------+                                                    |
|  |  Red Neuronal  |                                                    |
|  |  (theta)       |                                                    |
|  +----------------+                                                    |
|      |                                                                 |
|      v                                                                 |
|  [logits para cada accion]                                             |
|      |                                                                 |
|      v                                                                 |
|  SOFTMAX                                                               |
|      |                                                                 |
|      v                                                                 |
|  [P(a1|s), P(a2|s), P(a3|s), ...]  <- Probabilidades                   |
|      |                                                                 |
|      v                                                                 |
|  Sample accion segun probabilidades                                    |
|                                                                        |
|                                                                        |
|  Para acciones CONTINUAS:                                              |
|  -------------------------                                             |
|                                                                        |
|  Estado s                                                              |
|      |                                                                 |
|      v                                                                 |
|  +----------------+                                                    |
|  |  Red Neuronal  |                                                    |
|  |  (theta)       |                                                    |
|  +----------------+                                                    |
|      |         |                                                       |
|      v         v                                                       |
|    mu(s)    sigma(s)   <- Media y desviacion estandar                  |
|      |         |                                                       |
|      +----+----+                                                       |
|           |                                                            |
|           v                                                            |
|     Normal(mu, sigma)                                                  |
|           |                                                            |
|           v                                                            |
|     Accion continua (sample)                                           |
|                                                                        |
+------------------------------------------------------------------------+
```

## 2. Policy Gradient Theorem

### El Objetivo

```
+------------------------------------------------------------------------+
|  OBJETIVO: Maximizar retorno esperado                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  J(theta) = E_tau~pi_theta [R(tau)]                                    |
|                                                                        |
|  donde:                                                                |
|    tau = (s0, a0, r0, s1, a1, r1, ...) es una trayectoria              |
|    R(tau) = sum_t gamma^t * r_t  es el retorno                         |
|    pi_theta = politica parametrizada                                   |
|                                                                        |
|                                                                        |
|  Queremos encontrar theta* = argmax_theta J(theta)                     |
|                                                                        |
|  PROBLEMA: Como calcular gradiente de J(theta)?                        |
|            R(tau) depende de acciones que dependen de theta            |
|            pero tambien de transiciones del entorno                    |
|                                                                        |
+------------------------------------------------------------------------+
```

### El Teorema

```
+------------------------------------------------------------------------+
|  POLICY GRADIENT THEOREM                                               |
+------------------------------------------------------------------------+
|                                                                        |
|  nabla_theta J(theta) = E_tau~pi_theta [ sum_t nabla_theta log pi(a_t|s_t; theta) * R(tau) ]
|                                                                        |
|  DERIVACION SIMPLIFICADA:                                              |
|  -------------------------                                             |
|                                                                        |
|  J(theta) = E_tau [R(tau)]                                             |
|           = sum_tau P(tau; theta) * R(tau)                             |
|                                                                        |
|  nabla J = sum_tau nabla P(tau; theta) * R(tau)                        |
|                                                                        |
|  Usando el "log-derivative trick":                                     |
|    nabla P = P * nabla log P                                           |
|                                                                        |
|  nabla J = sum_tau P(tau; theta) * nabla log P(tau; theta) * R(tau)    |
|          = E_tau [ nabla log P(tau; theta) * R(tau) ]                  |
|                                                                        |
|  Ahora, P(tau; theta) = P(s0) * prod_t pi(a_t|s_t; theta) * P(s_t+1|s_t,a_t)
|                                                                        |
|  log P(tau; theta) = log P(s0) + sum_t [log pi(a_t|s_t) + log P(s_t+1|s_t,a_t)]
|                                                                        |
|  nabla log P(tau; theta) = sum_t nabla log pi(a_t|s_t; theta)          |
|                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^         |
|                            Solo la politica depende de theta!          |
|                                                                        |
|  RESULTADO FINAL:                                                      |
|                                                                        |
|  nabla J(theta) = E_tau [ sum_t nabla_theta log pi(a_t|s_t; theta) * R(tau) ]
|                                                                        |
|                                                                        |
|  INTUICION:                                                            |
|  ----------                                                            |
|  - log pi(a|s) es el "log-probability" de tomar accion a en estado s   |
|  - Si R(tau) es alto, aumentar probabilidad de esas acciones           |
|  - Si R(tau) es bajo, disminuir probabilidad de esas acciones          |
|                                                                        |
+------------------------------------------------------------------------+
```

## 3. REINFORCE Algorithm

### Algoritmo Basico

```
+------------------------------------------------------------------------+
|  ALGORITMO REINFORCE (Williams, 1992)                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  El algoritmo policy gradient mas simple                               |
|                                                                        |
|  Inicializar politica pi(a|s; theta) con pesos aleatorios              |
|                                                                        |
|  Para cada episodio:                                                   |
|      |                                                                 |
|      |  1. Generar trayectoria: tau = (s0,a0,r0, s1,a1,r1, ..., sT)    |
|      |     siguiendo pi(a|s; theta)                                    |
|      |                                                                 |
|      |  2. Para cada paso t = 0, 1, ..., T-1:                          |
|      |     Calcular retorno-to-go: G_t = sum_{k=t}^T gamma^(k-t) * r_k |
|      |                                                                 |
|      |  3. Actualizar theta:                                           |
|      |     theta <- theta + alpha * sum_t [nabla log pi(a_t|s_t) * G_t]|
|      |                                                                 |
|                                                                        |
|                                                                        |
|  NOTA: G_t (return-to-go) en lugar de R(tau) completo                  |
|        porque acciones futuras no afectan recompensas pasadas          |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion REINFORCE

```python
"""
REINFORCE - Policy Gradient basico.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List, Tuple
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    """
    Red de politica para acciones discretas.
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
        """Retorna logits (antes de softmax)."""
        return self.network(state)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """Retorna distribucion categorica sobre acciones."""
        logits = self.forward(state)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Sample accion de la politica.

        Returns:
            (accion, log_prob)
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class REINFORCEAgent:
    """
    Agente REINFORCE.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99
    ):
        self.gamma = gamma

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Almacenar trayectoria del episodio
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """Selecciona accion y guarda log_prob."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)

        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float) -> None:
        """Guarda recompensa."""
        self.rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """
        Calcula returns-to-go descontados.

        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        """
        returns = []
        G = 0

        # Calcular de atras hacia adelante
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalizar returns (reduce varianza)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self) -> float:
        """
        Actualiza la politica usando REINFORCE.

        Returns:
            Loss value
        """
        returns = self.compute_returns()

        # Policy gradient loss
        # Loss = -E[log pi(a|s) * G]  (negativo porque optimizamos minimizando)
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Limpiar buffers
        self.log_probs = []
        self.rewards = []

        return loss.item()


def train_reinforce(
    env_name: str = "CartPole-v1",
    n_episodes: int = 1000,
    max_steps: int = 500
) -> Tuple[REINFORCEAgent, List[float]]:
    """
    Entrena agente REINFORCE.
    """
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99
    )

    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        # Update al final del episodio
        loss = agent.update()

        rewards_history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Loss: {loss:.4f}")

    env.close()
    return agent, rewards_history


if __name__ == "__main__":
    print("=== Entrenando REINFORCE en CartPole ===\n")
    agent, rewards = train_reinforce(n_episodes=1000)

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('REINFORCE - Curva de Aprendizaje')
    plt.grid(True, alpha=0.3)
    plt.savefig('reinforce_training.png', dpi=150)
    plt.show()
```

## 4. Reduciendo Varianza: Baselines

### El Problema de la Varianza

```
+------------------------------------------------------------------------+
|  PROBLEMA: ALTA VARIANZA EN REINFORCE                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  El gradiente estimado tiene alta varianza:                            |
|                                                                        |
|  nabla J = E[ sum_t nabla log pi(a_t|s_t) * G_t ]                      |
|                                        ^^^                             |
|                                   G_t puede variar mucho               |
|                                                                        |
|                                                                        |
|  Ejemplo:                                                              |
|  - Episodio 1: G_t = 100  -> "subir probabilidad de estas acciones"    |
|  - Episodio 2: G_t = 98   -> "subir probabilidad de estas acciones"    |
|  - Episodio 3: G_t = 102  -> "subir probabilidad de estas acciones"    |
|                                                                        |
|  Todos los episodios suben probabilidad, aunque algunos son peores!    |
|                                                                        |
|                                                                        |
|  SOLUCION: Restar una BASELINE                                         |
|  -----------------------------                                         |
|                                                                        |
|  nabla J = E[ sum_t nabla log pi(a_t|s_t) * (G_t - b(s_t)) ]           |
|                                             ^^^^^^^^^^^^^^^            |
|                                             Advantage                  |
|                                                                        |
|  - Si G_t > b: accion mejor que promedio -> subir probabilidad         |
|  - Si G_t < b: accion peor que promedio -> bajar probabilidad          |
|                                                                        |
|  La baseline NO cambia el valor esperado del gradiente,                |
|  pero REDUCE la varianza significativamente.                           |
|                                                                        |
+------------------------------------------------------------------------+
```

### REINFORCE con Baseline

```python
"""
REINFORCE con Baseline (Value Network).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List, Tuple


class PolicyValueNetwork(nn.Module):
    """
    Red combinada para politica y valor (share some layers).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Capas compartidas
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Cabeza de politica
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Cabeza de valor (baseline)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            (policy_logits, value)
        """
        shared_features = self.shared(state)
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy_logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample accion y obtener valor.

        Returns:
            (accion, log_prob, value)
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()


class REINFORCEBaselineAgent:
    """
    REINFORCE con baseline aprendida.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        gamma: float = 0.99
    ):
        self.gamma = gamma

        self.network = PolicyValueNetwork(state_dim, action_dim, hidden_dim)

        # Optimizadores separados (o uno combinado)
        self.optimizer = optim.Adam([
            {'params': self.network.shared.parameters(), 'lr': lr_policy},
            {'params': self.network.policy_head.parameters(), 'lr': lr_policy},
            {'params': self.network.value_head.parameters(), 'lr': lr_value}
        ])

        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """Selecciona accion y guarda log_prob y value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = self.network.get_action_and_value(state_tensor)

        self.log_probs.append(log_prob)
        self.values.append(value)
        return action

    def store_reward(self, reward: float) -> None:
        """Guarda recompensa."""
        self.rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """Calcula returns-to-go."""
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self) -> Tuple[float, float]:
        """
        Actualiza politica y baseline.

        Returns:
            (policy_loss, value_loss)
        """
        returns = self.compute_returns()
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)

        # Advantage = Return - Baseline (value)
        advantages = returns - values.detach()

        # Normalizar advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss: -E[log pi * Advantage]
        policy_loss = -(log_probs * advantages).sum()

        # Value loss: MSE entre value predictions y returns
        value_loss = nn.functional.mse_loss(values, returns)

        # Loss total
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Limpiar
        self.log_probs = []
        self.values = []
        self.rewards = []

        return policy_loss.item(), value_loss.item()


def train_reinforce_baseline(
    env_name: str = "CartPole-v1",
    n_episodes: int = 1000
) -> Tuple[REINFORCEBaselineAgent, List[float]]:
    """Entrena REINFORCE con baseline."""
    env = gym.make(env_name)

    agent = REINFORCEBaselineAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        policy_loss, value_loss = agent.update()
        rewards_history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1} | Avg Reward: {avg:.2f} | "
                  f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")

    env.close()
    return agent, rewards_history
```

## 5. Proximal Policy Optimization (PPO)

### Problemas con Policy Gradient Vanilla

```
+------------------------------------------------------------------------+
|  PROBLEMAS CON POLICY GRADIENT BASICO                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  1. STEP SIZE DIFICIL DE ELEGIR                                        |
|     ----------------------------                                       |
|     - Paso muy pequeno: aprendizaje muy lento                          |
|     - Paso muy grande: puede destruir la politica                      |
|                                                                        |
|     Un mal update puede colapsar el rendimiento:                       |
|                                                                        |
|     Rendimiento                                                        |
|         |        ____                                                  |
|         |       /    \                                                 |
|         |      /      \                                                |
|         |_____/        \_____  <- Update muy grande                    |
|         |                                                              |
|         +------------------------ Updates                              |
|                                                                        |
|                                                                        |
|  2. DATA INEFFICIENCY                                                  |
|     ------------------                                                 |
|     - Cada batch de datos se usa UNA vez                               |
|     - On-policy: no podemos reusar datos viejos                        |
|                                                                        |
|                                                                        |
|  PPO SOLUCIONA AMBOS PROBLEMAS                                         |
|                                                                        |
+------------------------------------------------------------------------+
```

### TRPO: La Idea Original

```
+------------------------------------------------------------------------+
|  TRUST REGION POLICY OPTIMIZATION (TRPO)                               |
+------------------------------------------------------------------------+
|                                                                        |
|  IDEA: Limitar cuanto cambia la politica en cada update                |
|                                                                        |
|  Optimizar:                                                            |
|    max_theta  L(theta)                                                 |
|                                                                        |
|  Sujeto a:                                                             |
|    KL(pi_old || pi_new) <= delta                                       |
|    ^^^^^^^^^^^^^^^^^^^^^^^                                             |
|    Divergencia KL limitada                                             |
|                                                                        |
|                                                                        |
|  L(theta) = E[ pi(a|s;theta)/pi(a|s;theta_old) * A(s,a) ]              |
|                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                     |
|                Importance sampling ratio                               |
|                                                                        |
|                                                                        |
|  PROBLEMA: TRPO es complicado de implementar                           |
|    - Requiere calcular Hessiano                                        |
|    - Conjugate gradient para resolver constraint                       |
|    - Computacionalmente costoso                                        |
|                                                                        |
+------------------------------------------------------------------------+
```

### PPO: La Version Practica

```
+------------------------------------------------------------------------+
|  PROXIMAL POLICY OPTIMIZATION (PPO)                                    |
+------------------------------------------------------------------------+
|                                                                        |
|  PPO aproxima TRPO de forma simple y efectiva.                         |
|                                                                        |
|  Dos versiones principales:                                            |
|                                                                        |
|                                                                        |
|  1. PPO-CLIP (la mas usada):                                           |
|     ------------------------                                           |
|                                                                        |
|     r_t(theta) = pi(a|s;theta) / pi(a|s;theta_old)                     |
|                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                         |
|                  Probability ratio                                     |
|                                                                        |
|     L^CLIP = min(                                                      |
|         r_t * A_t,                          <- Objetivo normal         |
|         clip(r_t, 1-eps, 1+eps) * A_t       <- Objetivo clipeado       |
|     )                                                                  |
|                                                                        |
|     eps tipico = 0.2                                                   |
|                                                                        |
|                                                                        |
|     INTUICION del clip:                                                |
|     -------------------                                                |
|     Si A > 0 (buena accion):                                           |
|       - Queremos aumentar pi(a|s)                                      |
|       - Pero no mas alla de (1+eps) * pi_old                           |
|                                                                        |
|     Si A < 0 (mala accion):                                            |
|       - Queremos disminuir pi(a|s)                                     |
|       - Pero no mas alla de (1-eps) * pi_old                           |
|                                                                        |
|                                                                        |
|  2. PPO-PENALTY:                                                       |
|     ------------                                                       |
|     L = L^policy - beta * KL(pi_old, pi_new)                           |
|                    ^^^^^^^^^^^^^^^^^^^^^^^^^^                          |
|                    Penalizacion adaptativa                             |
|                                                                        |
+------------------------------------------------------------------------+
```

### Algoritmo PPO

```
+------------------------------------------------------------------------+
|  ALGORITMO PPO                                                         |
+------------------------------------------------------------------------+
|                                                                        |
|  Para cada iteracion:                                                  |
|                                                                        |
|    1. Recolectar T timesteps de experiencia con pi_old                 |
|       (multiples episodios parciales en paralelo)                      |
|                                                                        |
|    2. Calcular advantages A_t usando GAE (Generalized Advantage Est.)  |
|                                                                        |
|    3. Optimizar L^CLIP + c1*L^VF - c2*S[pi]                            |
|       sobre K epochs, con minibatches                                  |
|                                                                        |
|       donde:                                                           |
|         L^CLIP = objective de politica clipeado                        |
|         L^VF   = (V(s) - V_target)^2  (value function loss)            |
|         S[pi]  = entropia de la politica (fomenta exploracion)         |
|         c1, c2 = coeficientes                                          |
|                                                                        |
|    4. theta_old <- theta                                               |
|                                                                        |
|                                                                        |
|  HIPERPARAMETROS TIPICOS:                                              |
|    - clip_epsilon = 0.2                                                |
|    - epochs = 4-10                                                     |
|    - minibatch_size = 64-256                                           |
|    - learning_rate = 3e-4                                              |
|    - gamma = 0.99                                                      |
|    - GAE lambda = 0.95                                                 |
|                                                                        |
+------------------------------------------------------------------------+
```

### Generalized Advantage Estimation (GAE)

```
+------------------------------------------------------------------------+
|  GAE: ESTIMACION DE VENTAJA GENERALIZADA                               |
+------------------------------------------------------------------------+
|                                                                        |
|  PROBLEMA: Como calcular A(s,a)?                                       |
|                                                                        |
|  Opciones simples:                                                     |
|    1. A_t = G_t - V(s_t)           <- Alta varianza                    |
|    2. A_t = r_t + gamma*V(s_{t+1}) - V(s_t)  <- Bajo bias pero sesgado |
|                                                                        |
|                                                                        |
|  GAE: Interpolacion entre ambas                                        |
|  -------------------------------                                       |
|                                                                        |
|  delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)   <- TD error               |
|                                                                        |
|  A^GAE_t = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...
|          = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}              |
|                                                                        |
|                                                                        |
|  lambda controla el tradeoff bias-varianza:                            |
|    lambda = 0: A_t = delta_t (bajo varianza, alto bias)                |
|    lambda = 1: A_t = G_t - V(s_t) (alto varianza, bajo bias)           |
|    lambda = 0.95: balance tipico                                       |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion PPO Completa

```python
"""
PPO (Proximal Policy Optimization) implementacion completa.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass


class RolloutBuffer(NamedTuple):
    """Buffer para almacenar rollouts."""
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ActorCritic(nn.Module):
    """
    Red Actor-Critic para PPO.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False
    ):
        super().__init__()
        self.continuous = continuous

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Inicializacion
        self._init_weights()

    def _init_weights(self):
        """Inicializacion ortogonal (mejor para RL)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.features(state)
        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            return mean, value
        else:
            logits = self.actor(features)
            return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Obtiene accion, log_prob, entropia y valor.

        Si action es None, samplea nueva accion.
        Si action es proporcionada, calcula log_prob de esa accion.
        """
        features = self.features(state)
        value = self.critic(features).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value


class PPOAgent:
    """
    Agente PPO completo.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        continuous: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.continuous = continuous

        self.network = ActorCritic(
            state_dim, action_dim, hidden_dim, continuous
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(
        self,
        state: np.ndarray
    ) -> Tuple[np.ndarray | int, float, float]:
        """
        Selecciona accion.

        Returns:
            (accion, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)

        if self.continuous:
            return action.cpu().numpy()[0], log_prob.item(), value.item()
        else:
            return action.item(), log_prob.item(), value.item()

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula GAE y returns.

        Returns:
            (advantages, returns)
        """
        advantages = []
        gae = 0

        # Anadir next_value para bootstrap
        values = values + [next_value]

        # Calcular GAE de atras hacia adelante
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> dict:
        """
        Actualiza la red usando PPO-Clip.

        Returns:
            Diccionario con metricas de entrenamiento
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device) if not self.continuous \
            else torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalizar advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Indices para minibatches
        n_samples = len(states)
        indices = np.arange(n_samples)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Obtener batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )

                # Ratio para PPO
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence para early stopping
                approx_kl = ((ratio - 1) - log_ratio).mean()

                # Clipped objective
                policy_loss_1 = ratio * batch_advantages
                policy_loss_2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss (clipped tambien es opcion)
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus (fomenta exploracion)
                entropy_loss = -entropy.mean()

                # Loss total
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates
        }


def train_ppo(
    env_name: str = "CartPole-v1",
    n_iterations: int = 200,
    n_steps: int = 2048,
    n_envs: int = 1
) -> Tuple[PPOAgent, List[float]]:
    """
    Entrena agente PPO.
    """
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        continuous=continuous,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64
    )

    rewards_history = []
    episode_rewards = []
    current_reward = 0

    state, _ = env.reset()

    for iteration in range(n_iterations):
        # Buffers para esta iteracion
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        # Recolectar n_steps de experiencia
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            current_reward += reward
            state = next_state

            if done:
                episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = env.reset()

        # Bootstrap value para ultimo estado
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            _, _, _, next_value = agent.network.get_action_and_value(state_tensor)
            next_value = next_value.item()

        # Calcular GAE
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # Convertir a arrays
        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)

        # Update PPO
        metrics = agent.update(states, actions, log_probs, advantages, returns)

        # Log
        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-10:])
            rewards_history.extend(episode_rewards)
            episode_rewards = []

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Policy Loss: {metrics['policy_loss']:.4f} | "
                      f"Value Loss: {metrics['value_loss']:.4f} | "
                      f"KL: {metrics['approx_kl']:.4f}")

    env.close()
    return agent, rewards_history


if __name__ == "__main__":
    print("=== Entrenando PPO en CartPole ===\n")
    agent, rewards = train_ppo(n_iterations=200)

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    if len(rewards) >= 50:
        plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('PPO - Curva de Aprendizaje')
    plt.grid(True, alpha=0.3)
    plt.savefig('ppo_training.png', dpi=150)
    plt.show()
```

## 6. PPO para Acciones Continuas

```python
"""
PPO para entornos con acciones continuas.
Ejemplo: Pendulum, MountainCarContinuous, etc.
"""
import torch
import gymnasium as gym
import numpy as np


def train_ppo_continuous(
    env_name: str = "Pendulum-v1",
    n_iterations: int = 300,
    n_steps: int = 2048
):
    """Entrena PPO en entorno continuo."""
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        continuous=True,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10
    )

    rewards_history = []
    episode_rewards = []
    current_reward = 0

    state, _ = env.reset()

    for iteration in range(n_iterations):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(state)

            # Clip accion al rango valido
            action_clipped = np.clip(
                action,
                env.action_space.low,
                env.action_space.high
            )

            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            current_reward += reward
            state = next_state

            if done:
                episode_rewards.append(current_reward)
                current_reward = 0
                state, _ = env.reset()

        # Bootstrap
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            _, _, _, next_value = agent.network.get_action_and_value(state_tensor)
            next_value = next_value.item()

        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)

        metrics = agent.update(states, actions, log_probs, advantages, returns)

        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-10:])
            rewards_history.extend(episode_rewards)
            episode_rewards = []

            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1} | Avg Reward: {avg_reward:.2f}")

    env.close()
    return agent, rewards_history
```

## 7. Usando Stable-Baselines3

```python
"""
PPO con Stable-Baselines3 - la forma rapida y robusta.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def train_ppo_sb3(
    env_name: str = "CartPole-v1",
    total_timesteps: int = 100_000,
    n_envs: int = 4
):
    """
    Entrena PPO usando Stable-Baselines3.
    """
    # Crear vectorized environment (entrenamiento paralelo)
    env = make_vec_env(env_name, n_envs=n_envs)

    # Crear modelo PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    # Callbacks para evaluacion
    eval_env = gym.make(env_name)
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=495,  # Para CartPole
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,
        best_model_save_path="./ppo_best/",
        verbose=1
    )

    # Entrenar
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    # Guardar modelo
    model.save("ppo_cartpole")

    # Evaluar
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10
    )
    print(f"\nEvaluacion: {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_env.close()
    env.close()

    return model


def demo_trained_agent(model_path: str, env_name: str, n_episodes: int = 5):
    """Demostrar agente entrenado."""
    model = PPO.load(model_path)
    env = gym.make(env_name, render_mode="human")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    print("=== Entrenando PPO con Stable-Baselines3 ===\n")
    model = train_ppo_sb3(total_timesteps=100_000)

    print("\n=== Demostracion del agente ===")
    demo_trained_agent("ppo_cartpole", "CartPole-v1", n_episodes=3)
```

## 8. Aplicacion: Agente de Pentesting con PPO

```python
"""
Agente de penetration testing usando PPO.

El agente aprende a explorar y explotar vulnerabilidades
en un entorno de red simulado.
"""
import numpy as np
import torch
from typing import Tuple, List, Set
from dataclasses import dataclass, field
from enum import Enum


class PentestAction(Enum):
    """Acciones del pentester."""
    NMAP_SCAN = 0
    VULN_SCAN = 1
    EXPLOIT_SSH = 2
    EXPLOIT_WEB = 3
    EXPLOIT_SMB = 4
    PRIVESC = 5
    PIVOT = 6
    EXFIL = 7


@dataclass
class NetworkNode:
    """Nodo en la red."""
    ip: str
    hostname: str
    os: str
    services: Set[str] = field(default_factory=set)
    vulnerabilities: Set[str] = field(default_factory=set)
    access_level: int = 0  # 0=none, 1=user, 2=admin, 3=root
    has_flag: bool = False
    discovered: bool = False


class PentestEnvironment:
    """
    Entorno de pentesting para RL.

    Simula una red corporativa con varios hosts y vulnerabilidades.
    """

    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.nodes: dict[str, NetworkNode] = {}
        self.current_node_ip: str = ""
        self.flags_captured: int = 0
        self.total_flags: int = 0
        self.steps: int = 0
        self.max_steps: int = 200
        self.detected: bool = False

        self._setup_network()

    def _setup_network(self) -> None:
        """Configura la red objetivo."""
        # DMZ
        self.nodes["10.0.1.10"] = NetworkNode(
            ip="10.0.1.10",
            hostname="web01",
            os="Linux",
            services={"http", "https", "ssh"},
            vulnerabilities={"CVE-2021-44228", "weak_ssh_config"},
            has_flag=True
        )

        # Internal network
        self.nodes["192.168.1.50"] = NetworkNode(
            ip="192.168.1.50",
            hostname="fileserver",
            os="Windows",
            services={"smb", "netbios"},
            vulnerabilities={"EternalBlue", "weak_smb_signing"},
            has_flag=True
        )

        self.nodes["192.168.1.100"] = NetworkNode(
            ip="192.168.1.100",
            hostname="dc01",
            os="Windows Server",
            services={"ldap", "kerberos", "smb"},
            vulnerabilities={"ZeroLogon"},
            has_flag=True
        )

        self.nodes["192.168.1.200"] = NetworkNode(
            ip="192.168.1.200",
            hostname="db01",
            os="Linux",
            services={"mysql", "ssh"},
            vulnerabilities={"mysql_unauth"},
            has_flag=True
        )

        self.total_flags = sum(1 for n in self.nodes.values() if n.has_flag)

        # Punto de entrada
        self.nodes["10.0.1.10"].discovered = True
        self.current_node_ip = "10.0.1.10"

    @property
    def state_dim(self) -> int:
        """Dimension del estado."""
        return 4 * 8 + 5  # 4 nodos * 8 features + 5 globales

    @property
    def action_dim(self) -> int:
        """Numero de acciones."""
        return len(PentestAction)

    def _get_state(self) -> np.ndarray:
        """Convierte estado del entorno a vector."""
        state = []

        # Features por nodo
        for ip in sorted(self.nodes.keys()):
            node = self.nodes[ip]
            node_state = [
                1.0 if node.discovered else 0.0,
                len(node.services) / 5,
                len(node.vulnerabilities) / 5,
                node.access_level / 3,
                1.0 if node.has_flag and node.access_level >= 2 else 0.0,
                1.0 if "ssh" in node.services else 0.0,
                1.0 if "http" in node.services else 0.0,
                1.0 if "smb" in node.services else 0.0
            ]
            state.extend(node_state)

        # Estado global
        state.extend([
            self.flags_captured / self.total_flags,
            self.steps / self.max_steps,
            1.0 if self.detected else 0.0,
            sum(1 for n in self.nodes.values() if n.discovered) / len(self.nodes),
            sum(1 for n in self.nodes.values() if n.access_level > 0) / len(self.nodes)
        ])

        return np.array(state, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reinicia entorno."""
        self._setup_network()
        self.flags_captured = 0
        self.steps = 0
        self.detected = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta accion.

        Returns:
            (nuevo_estado, recompensa, done, info)
        """
        self.steps += 1
        action = PentestAction(action)
        current_node = self.nodes[self.current_node_ip]

        reward = -0.1  # Pequeno coste por paso
        info = {"action": action.name, "result": "none"}

        # Probabilidad de deteccion (aumenta con acciones ruidosas)
        detection_prob = {
            PentestAction.NMAP_SCAN: 0.05,
            PentestAction.VULN_SCAN: 0.10,
            PentestAction.EXPLOIT_SSH: 0.15,
            PentestAction.EXPLOIT_WEB: 0.10,
            PentestAction.EXPLOIT_SMB: 0.20,
            PentestAction.PRIVESC: 0.15,
            PentestAction.PIVOT: 0.05,
            PentestAction.EXFIL: 0.25
        }

        if np.random.random() < detection_prob[action]:
            self.detected = True
            reward = -50
            info["result"] = "DETECTED"
            return self._get_state(), reward, True, info

        # Ejecutar accion
        if action == PentestAction.NMAP_SCAN:
            # Descubrir nuevos hosts
            for ip, node in self.nodes.items():
                if not node.discovered and np.random.random() > 0.5:
                    node.discovered = True
                    reward += 5
                    info["result"] = f"discovered_{ip}"

        elif action == PentestAction.VULN_SCAN:
            # Revelar vulnerabilidades (ya las tenemos, pero simular scan)
            if current_node.vulnerabilities:
                reward += 2
                info["result"] = f"vulns_found_{len(current_node.vulnerabilities)}"

        elif action == PentestAction.EXPLOIT_SSH:
            if "weak_ssh_config" in current_node.vulnerabilities:
                if current_node.access_level == 0:
                    current_node.access_level = 1
                    reward += 15
                    info["result"] = "ssh_shell_obtained"

        elif action == PentestAction.EXPLOIT_WEB:
            if "CVE-2021-44228" in current_node.vulnerabilities:
                if current_node.access_level == 0:
                    current_node.access_level = 1
                    reward += 15
                    info["result"] = "web_shell_obtained"

        elif action == PentestAction.EXPLOIT_SMB:
            if "EternalBlue" in current_node.vulnerabilities:
                if current_node.access_level == 0:
                    current_node.access_level = 2
                    reward += 20
                    info["result"] = "smb_admin_obtained"

        elif action == PentestAction.PRIVESC:
            if current_node.access_level == 1:
                # Intentar privesc
                if np.random.random() > 0.4:  # 60% exito
                    current_node.access_level = 3
                    reward += 20
                    info["result"] = "root_obtained"

                    # Capturar flag si existe
                    if current_node.has_flag:
                        self.flags_captured += 1
                        reward += 50
                        info["result"] = "FLAG_CAPTURED"

        elif action == PentestAction.PIVOT:
            # Moverse a otro nodo descubierto con acceso
            targets = [
                ip for ip, n in self.nodes.items()
                if n.discovered and ip != self.current_node_ip
            ]
            if targets:
                self.current_node_ip = np.random.choice(targets)
                reward += 3
                info["result"] = f"pivoted_to_{self.current_node_ip}"

        elif action == PentestAction.EXFIL:
            if current_node.access_level >= 2:
                reward += 10
                info["result"] = "data_exfiltrated"

        # Check fin del episodio
        done = (
            self.steps >= self.max_steps or
            self.flags_captured >= self.total_flags or
            self.detected
        )

        if self.flags_captured >= self.total_flags:
            reward += 100
            info["result"] = "ALL_FLAGS_CAPTURED"

        return self._get_state(), reward, done, info


def train_pentest_agent(n_iterations: int = 500) -> Tuple[PPOAgent, List[float]]:
    """Entrena agente de pentesting."""
    env = PentestEnvironment()

    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        continuous=False,
        lr=3e-4,
        gamma=0.99,
        n_epochs=10
    )

    rewards_history = []

    for iteration in range(n_iterations):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        state = env.reset()
        episode_reward = 0

        while True:
            action, log_prob, value = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            episode_reward += reward
            state = next_state

            if done:
                break

        rewards_history.append(episode_reward)

        # Bootstrap
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            _, _, _, next_value = agent.network.get_action_and_value(state_tensor)
            next_value = 0 if done else next_value.item()

        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)

        metrics = agent.update(states, actions, log_probs, advantages, returns)

        if (iteration + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Iteration {iteration + 1} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Flags: {env.flags_captured}/{env.total_flags}")

    return agent, rewards_history


if __name__ == "__main__":
    print("=== Entrenando Agente Pentester con PPO ===\n")
    agent, rewards = train_pentest_agent(n_iterations=500)

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Pentest Agent PPO - Curva de Aprendizaje')
    plt.grid(True, alpha=0.3)
    plt.savefig('pentest_ppo_training.png', dpi=150)
    plt.show()
```

## 9. Resumen

```
+------------------------------------------------------------------------+
|  POLICY GRADIENT - RESUMEN                                             |
+------------------------------------------------------------------------+
|                                                                        |
|  POLICY GRADIENT THEOREM:                                              |
|    nabla J = E[nabla log pi(a|s) * return]                             |
|    Aumentar prob de acciones con alto return                           |
|                                                                        |
|  REINFORCE:                                                            |
|    - Algoritmo mas simple                                              |
|    - Alta varianza (problema)                                          |
|    - Baseline reduce varianza (V(s) como baseline)                     |
|                                                                        |
|  TRPO:                                                                 |
|    - Limita cambio en politica (KL constraint)                         |
|    - Estable pero complejo de implementar                              |
|                                                                        |
|  PPO:                                                                  |
|    - Version practica de TRPO                                          |
|    - Clip objective: limita ratio pi_new/pi_old                        |
|    - Mas sample efficient (multiples epochs por batch)                 |
|    - El algoritmo mas usado en la practica                             |
|                                                                        |
|  GAE (Generalized Advantage Estimation):                               |
|    - Balance bias-varianza en estimacion de advantage                  |
|    - lambda controla el tradeoff                                       |
|                                                                        |
|  VENTAJAS POLICY GRADIENT:                                             |
|    - Acciones continuas nativas                                        |
|    - Politicas estocasticas                                            |
|    - Convergencia mas suave                                            |
|                                                                        |
|  STABLE-BASELINES3:                                                    |
|    - Implementacion robusta y facil de usar                            |
|    - Soporte para muchos algoritmos (PPO, A2C, SAC, etc.)              |
|    - Vectorized environments para paralelismo                          |
|                                                                        |
+------------------------------------------------------------------------+
```

---

**Siguiente:** Actor-Critic Methods (A2C, A3C, SAC)
