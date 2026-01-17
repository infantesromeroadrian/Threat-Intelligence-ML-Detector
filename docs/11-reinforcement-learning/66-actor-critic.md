# Actor-Critic Methods

## 1. Introduccion a Actor-Critic

### Combinando Value-Based y Policy-Based

```
+------------------------------------------------------------------------+
|  ACTOR-CRITIC: LO MEJOR DE DOS MUNDOS                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  VALUE-BASED:                      POLICY-BASED:                       |
|  - Aprende Q(s,a) o V(s)           - Aprende pi(a|s) directamente      |
|  - Politica derivada (argmax)      - Puede ser estocastica             |
|  - Bajo varianza                   - Alta varianza                     |
|  - Solo acciones discretas         - Acciones continuas OK             |
|                                                                        |
|                                                                        |
|                     ACTOR-CRITIC                                       |
|                          |                                             |
|            +-------------+-------------+                               |
|            |                           |                               |
|       +--------+                  +--------+                           |
|       | ACTOR  |                  | CRITIC |                           |
|       +--------+                  +--------+                           |
|            |                           |                               |
|       pi(a|s)                      V(s) o Q(s,a)                        |
|       (politica)                   (valor)                             |
|            |                           |                               |
|            +-------------+-------------+                               |
|                          |                                             |
|                   Entrenamiento                                        |
|                    coordinado                                          |
|                                                                        |
|                                                                        |
|  FLUJO:                                                                |
|  ------                                                                |
|  1. Actor propone accion segun pi(a|s)                                 |
|  2. Entorno retorna reward y nuevo estado                              |
|  3. Critic evalua la accion (TD error)                                 |
|  4. Actor se actualiza usando feedback del Critic                      |
|  5. Critic se actualiza con reward real                                |
|                                                                        |
+------------------------------------------------------------------------+
```

### Arquitectura Detallada

```
+------------------------------------------------------------------------+
|  ARQUITECTURA ACTOR-CRITIC                                             |
+------------------------------------------------------------------------+
|                                                                        |
|              Estado s                                                  |
|                 |                                                      |
|      +----------+----------+                                           |
|      |                     |                                           |
|      v                     v                                           |
|  +--------+           +--------+                                       |
|  | ACTOR  |           | CRITIC |                                       |
|  | theta  |           |   w    |                                       |
|  +--------+           +--------+                                       |
|      |                     |                                           |
|      v                     v                                           |
|  pi(a|s; theta)       V(s; w) o Q(s,a; w)                              |
|      |                     |                                           |
|      v                     |                                           |
|  Sample accion a           |                                           |
|      |                     |                                           |
|      v                     |                                           |
|  Entorno                   |                                           |
|      |                     |                                           |
|      v                     v                                           |
|  r, s'  -----------------> TD Error = r + gamma*V(s') - V(s)           |
|                                |                                       |
|                                v                                       |
|                       Actualizar Critic                                |
|                       (minimizar TD error)                             |
|                                |                                       |
|                                v                                       |
|                       Actualizar Actor                                 |
|                       (usar TD error como ventaja)                     |
|                                                                        |
|                                                                        |
|  UPDATE RULES:                                                         |
|  -------------                                                         |
|  Critic: w <- w + alpha_w * delta * nabla_w V(s; w)                    |
|  Actor:  theta <- theta + alpha_theta * delta * nabla_theta log pi(a|s)|
|                                                                        |
|  donde delta = TD error = r + gamma*V(s') - V(s)                       |
|                                                                        |
+------------------------------------------------------------------------+
```

## 2. Advantage Actor-Critic (A2C)

### El Concepto de Advantage

```
+------------------------------------------------------------------------+
|  ADVANTAGE FUNCTION                                                    |
+------------------------------------------------------------------------+
|                                                                        |
|  A(s, a) = Q(s, a) - V(s)                                              |
|                                                                        |
|  INTERPRETACION:                                                       |
|  ---------------                                                       |
|  - Q(s,a): valor de tomar accion a en estado s                         |
|  - V(s): valor promedio de estar en s                                  |
|  - A(s,a): "ventaja" de a respecto al promedio                         |
|                                                                        |
|  A(s,a) > 0: accion MEJOR que promedio -> aumentar probabilidad        |
|  A(s,a) < 0: accion PEOR que promedio -> disminuir probabilidad        |
|  A(s,a) = 0: accion promedio                                           |
|                                                                        |
|                                                                        |
|  POR QUE USAR ADVANTAGE?                                               |
|  -----------------------                                               |
|                                                                        |
|  REINFORCE basico:                                                     |
|    nabla J = E[nabla log pi(a|s) * G_t]                                |
|                                  ^^^                                   |
|                              return alto                               |
|                              -> siempre positivo?                      |
|                              -> alta varianza                          |
|                                                                        |
|  Con Advantage:                                                        |
|    nabla J = E[nabla log pi(a|s) * A(s,a)]                             |
|                                  ^^^^^^^                               |
|                              puede ser negativo                        |
|                              -> centrado en 0                          |
|                              -> baja varianza                          |
|                                                                        |
+------------------------------------------------------------------------+
```

### Estimando Advantage con TD Error

```
+------------------------------------------------------------------------+
|  ESTIMACION DE ADVANTAGE CON TD ERROR                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  Problema: No conocemos Q(s,a) directamente                            |
|                                                                        |
|  Solucion: Usar TD error como estimador de advantage                   |
|                                                                        |
|                                                                        |
|  TD Error (1-step):                                                    |
|  ------------------                                                    |
|  delta = r + gamma * V(s') - V(s)                                      |
|                                                                        |
|  Esto es un estimador insesgado de A(s,a):                             |
|    E[delta] = E[r + gamma*V(s')] - V(s)                                |
|             = Q(s,a) - V(s)                                            |
|             = A(s,a)                                                   |
|                                                                        |
|                                                                        |
|  N-step TD:                                                            |
|  ----------                                                            |
|  delta_t^(n) = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1}       |
|              + gamma^n * V(s_{t+n}) - V(s_t)                           |
|                                                                        |
|  Tradeoff:                                                             |
|    - n pequeno: bajo varianza, alto bias                               |
|    - n grande: alto varianza, bajo bias                                |
|                                                                        |
|                                                                        |
|  GAE (Generalized Advantage Estimation):                               |
|  ---------------------------------------                               |
|  A^GAE = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}                |
|                                                                        |
|  lambda controla el tradeoff (ver PPO)                                 |
|                                                                        |
+------------------------------------------------------------------------+
```

### Algoritmo A2C

```
+------------------------------------------------------------------------+
|  ALGORITMO A2C (Advantage Actor-Critic)                                |
+------------------------------------------------------------------------+
|                                                                        |
|  Inicializar:                                                          |
|    - Actor pi(a|s; theta)                                              |
|    - Critic V(s; w)                                                    |
|                                                                        |
|  Para cada iteracion:                                                  |
|                                                                        |
|    1. Recolectar batch de experiencias con N workers en paralelo       |
|       Cada worker genera trayectoria de T pasos                        |
|                                                                        |
|    2. Calcular returns y advantages para cada transicion:              |
|       Para t = T-1, T-2, ..., 0:                                       |
|         delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)                    |
|         A_t = delta_t + gamma*lambda*A_{t+1}    (GAE)                  |
|         R_t = A_t + V(s_t)                      (return target)        |
|                                                                        |
|    3. Calcular loss:                                                   |
|       L_policy = -E[log pi(a|s) * A]        (policy gradient)          |
|       L_value  = E[(V(s) - R)^2]            (value regression)         |
|       L_entropy = -E[H(pi)]                 (entropy bonus)            |
|                                                                        |
|       L_total = L_policy + c1*L_value - c2*L_entropy                   |
|                                                                        |
|    4. Actualizar parametros con gradiente descendente                  |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion A2C

```python
"""
A2C (Advantage Actor-Critic) implementacion completa.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List, Tuple
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe
from collections import deque


class ActorCriticNetwork(nn.Module):
    """
    Red compartida Actor-Critic.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Feature extractor compartido
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Inicializacion
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        # Output layers con ganancia menor
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Obtiene accion, log_prob, entropia y valor.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class A2CAgent:
    """
    Agente A2C.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Selecciona accion."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)

        return action.item(), log_prob.item(), value.item()

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcula GAE y returns."""
        advantages = []
        gae = 0

        values = values + [next_value]

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
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> dict:
        """Actualiza la red."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalizar advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        _, log_probs, entropy, values = self.network.get_action_and_value(states, actions)

        # Losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item()
        }


def train_a2c(
    env_name: str = "CartPole-v1",
    n_updates: int = 1000,
    n_steps: int = 5,
    n_envs: int = 8
) -> Tuple[A2CAgent, List[float]]:
    """
    Entrena A2C con multiples entornos en paralelo.
    """
    # Crear entornos
    envs = [gym.make(env_name) for _ in range(n_envs)]

    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n

    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_steps=n_steps
    )

    # Estados iniciales
    states = np.array([env.reset()[0] for env in envs])

    rewards_history = []
    episode_rewards = [0] * n_envs
    completed_rewards = deque(maxlen=100)

    for update in range(n_updates):
        # Buffers para este rollout
        all_states = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_dones = []

        # Recolectar n_steps de experiencia
        for step in range(n_steps):
            all_states.append(states.copy())

            # Seleccionar acciones para todos los envs
            actions = []
            values = []
            for i, state in enumerate(states):
                action, _, value = agent.select_action(state)
                actions.append(action)
                values.append(value)

            all_actions.append(actions)
            all_values.append(values)

            # Ejecutar acciones
            next_states = []
            rewards = []
            dones = []

            for i, (env, action) in enumerate(zip(envs, actions)):
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_rewards[i] += reward

                if done:
                    completed_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    next_state, _ = env.reset()

                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

            all_rewards.append(rewards)
            all_dones.append(dones)
            states = np.array(next_states)

        # Bootstrap values
        next_values = []
        for state in states:
            _, _, value = agent.select_action(state)
            next_values.append(value)

        # Procesar cada entorno
        batch_states = []
        batch_actions = []
        batch_advantages = []
        batch_returns = []

        for env_idx in range(n_envs):
            env_rewards = [all_rewards[t][env_idx] for t in range(n_steps)]
            env_values = [all_values[t][env_idx] for t in range(n_steps)]
            env_dones = [all_dones[t][env_idx] for t in range(n_steps)]

            advantages, returns = agent.compute_gae(
                env_rewards, env_values, env_dones, next_values[env_idx]
            )

            for t in range(n_steps):
                batch_states.append(all_states[t][env_idx])
                batch_actions.append(all_actions[t][env_idx])

            batch_advantages.append(advantages)
            batch_returns.append(returns)

        # Concatenar batches
        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_advantages = torch.cat(batch_advantages)
        batch_returns = torch.cat(batch_returns)

        # Update
        metrics = agent.update(batch_states, batch_actions, batch_advantages, batch_returns)

        # Log
        if completed_rewards and (update + 1) % 50 == 0:
            avg_reward = np.mean(list(completed_rewards))
            rewards_history.append(avg_reward)
            print(f"Update {update + 1}/{n_updates} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Value Loss: {metrics['value_loss']:.4f}")

    # Cleanup
    for env in envs:
        env.close()

    return agent, rewards_history


if __name__ == "__main__":
    print("=== Entrenando A2C en CartPole ===\n")
    agent, rewards = train_a2c(n_updates=1000)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Update (x50)')
    plt.ylabel('Recompensa Media')
    plt.title('A2C - Curva de Aprendizaje')
    plt.grid(True, alpha=0.3)
    plt.savefig('a2c_training.png', dpi=150)
    plt.show()
```

## 3. A3C (Asynchronous Advantage Actor-Critic)

### La Idea Clave: Asincronismo

```
+------------------------------------------------------------------------+
|  A3C: ENTRENAMIENTO ASINCRONO                                          |
+------------------------------------------------------------------------+
|                                                                        |
|  PROBLEMA CON ENTRENAMIENTO SECUENCIAL:                                |
|  - Las muestras estan correlacionadas                                  |
|  - Un GPU/CPU subutilizado                                             |
|  - Entrenamiento lento                                                 |
|                                                                        |
|  DQN soluciona correlacion con Experience Replay                       |
|  A3C soluciona con PARALELISMO                                         |
|                                                                        |
|                                                                        |
|  ARQUITECTURA A3C:                                                     |
|  -----------------                                                     |
|                                                                        |
|         +-------------------+                                          |
|         | Global Network    |                                          |
|         | theta_global      |                                          |
|         +-------------------+                                          |
|           ^   ^   ^   ^                                                |
|           |   |   |   |    (gradientes)                                |
|           |   |   |   |                                                |
|    +------+   |   |   +------+                                         |
|    |          |   |          |                                         |
|  +----+   +----+   +----+   +----+                                     |
|  |W1  |   |W2  |   |W3  |   |W4  |    Workers independientes           |
|  |env1|   |env2|   |env3|   |env4|    (cada uno con su entorno)        |
|  +----+   +----+   +----+   +----+                                     |
|                                                                        |
|  Cada worker:                                                          |
|    1. Copia pesos globales -> theta_local                              |
|    2. Interactua con su entorno por n pasos                            |
|    3. Calcula gradientes locales                                       |
|    4. Aplica gradientes al modelo GLOBAL (asincronamente)              |
|    5. Repetir                                                          |
|                                                                        |
|                                                                        |
|  VENTAJAS:                                                             |
|  ---------                                                             |
|  - Datos decorrelacionados (diferentes entornos)                       |
|  - Exploracion diversa (cada worker explora diferente)                 |
|  - Paralelismo en CPU (no necesita GPU)                                |
|  - No necesita replay buffer (on-policy)                               |
|                                                                        |
+------------------------------------------------------------------------+
```

### A2C vs A3C

```
+------------------------------------------------------------------------+
|  A2C vs A3C                                                            |
+------------------------------------------------------------------------+
|                                                                        |
|  A3C (Asynchronous):                  A2C (Synchronous):               |
|  -------------------                  ------------------               |
|                                                                        |
|  - Updates asincronos                 - Updates sincronos              |
|  - Cada worker actualiza              - Todos los workers              |
|    cuando termina                       actualizan juntos              |
|                                                                        |
|  - Mas rapido en teoria               - Mas estable                    |
|  - Mas dificil de implementar         - Mas facil de implementar       |
|  - Gradientes stale posibles          - Gradientes frescos             |
|                                                                        |
|  A3C:                                 A2C:                             |
|  W1 ---> update                       W1 ---+                          |
|  W2 ------> update                    W2 ---+---> sync --> update      |
|  W3 -> update                         W3 ---+                          |
|  W4 ---------> update                 W4 ---+                          |
|                                                                        |
|  EN LA PRACTICA:                                                       |
|  - A2C con GPUs es tan rapido como A3C                                 |
|  - A2C es mas facil de debuggear                                       |
|  - A2C se usa mas comunmente                                           |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion A3C (Simplificada)

```python
"""
A3C simplificado usando multiprocessing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import Tuple


class SharedAdam(optim.Adam):
    """Adam optimizer con estado compartido entre procesos."""

    def __init__(self, params, lr=1e-4):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Share memory
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class A3CNetwork(nn.Module):
    """Red para A3C."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        return self.actor(features), self.critic(features)


def worker(
    worker_id: int,
    global_model: A3CNetwork,
    optimizer: SharedAdam,
    env_name: str,
    gamma: float,
    n_steps: int,
    max_episodes: int,
    result_queue: mp.Queue
):
    """Worker process para A3C."""
    torch.manual_seed(worker_id)

    # Crear entorno y modelo local
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    local_model = A3CNetwork(state_dim, action_dim)

    for episode in range(max_episodes):
        # Sincronizar con modelo global
        local_model.load_state_dict(global_model.state_dict())

        state, _ = env.reset()
        done = False
        episode_reward = 0

        states, actions, rewards, dones = [], [], [], []

        step = 0
        while not done:
            step += 1

            # Seleccionar accion
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, _ = local_model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            state = next_state

            # Update cada n_steps o al terminar
            if step % n_steps == 0 or done:
                # Calcular returns
                if done:
                    R = 0
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    _, value = local_model(state_tensor)
                    R = value.item()

                returns = []
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)

                returns = torch.tensor(returns, dtype=torch.float32)

                # Forward local
                states_tensor = torch.FloatTensor(np.array(states))
                actions_tensor = torch.stack(actions)

                logits, values = local_model(states_tensor)
                values = values.squeeze()

                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy()

                # Losses
                advantages = returns - values.detach()
                policy_loss = -(log_probs * advantages).mean()
                value_loss = nn.functional.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                # Backward y update global
                optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

                # Copiar gradientes al modelo global
                for local_param, global_param in zip(
                    local_model.parameters(),
                    global_model.parameters()
                ):
                    global_param._grad = local_param.grad.clone()

                optimizer.step()

                # Sincronizar
                local_model.load_state_dict(global_model.state_dict())

                # Limpiar buffers
                states, actions, rewards, dones = [], [], [], []

        result_queue.put((worker_id, episode, episode_reward))

    env.close()


def train_a3c(
    env_name: str = "CartPole-v1",
    n_workers: int = 4,
    max_episodes: int = 500,
    n_steps: int = 20,
    gamma: float = 0.99
):
    """Entrena A3C."""
    # Crear entorno temporal para obtener dimensiones
    temp_env = gym.make(env_name)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()

    # Modelo global compartido
    global_model = A3CNetwork(state_dim, action_dim)
    global_model.share_memory()

    # Optimizer compartido
    optimizer = SharedAdam(global_model.parameters(), lr=1e-4)

    # Queue para resultados
    result_queue = mp.Queue()

    # Crear y lanzar workers
    processes = []
    for worker_id in range(n_workers):
        p = mp.Process(
            target=worker,
            args=(
                worker_id, global_model, optimizer, env_name,
                gamma, n_steps, max_episodes // n_workers, result_queue
            )
        )
        p.start()
        processes.append(p)

    # Recolectar resultados
    rewards_history = []
    total_episodes = 0

    while total_episodes < max_episodes:
        try:
            worker_id, episode, reward = result_queue.get(timeout=60)
            rewards_history.append(reward)
            total_episodes += 1

            if total_episodes % 50 == 0:
                avg = np.mean(rewards_history[-50:])
                print(f"Episodes: {total_episodes} | Avg Reward: {avg:.2f}")

        except Exception:
            break

    # Esperar a que terminen todos
    for p in processes:
        p.join()

    return global_model, rewards_history


if __name__ == "__main__":
    mp.set_start_method('spawn')
    print("=== Entrenando A3C en CartPole ===\n")
    model, rewards = train_a3c(n_workers=4, max_episodes=1000)
```

## 4. Soft Actor-Critic (SAC)

### Maximo de Entropia RL

```
+------------------------------------------------------------------------+
|  MAXIMUM ENTROPY RL                                                    |
+------------------------------------------------------------------------+
|                                                                        |
|  RL ESTANDAR:                                                          |
|  Maximizar: E[sum_t gamma^t * r_t]                                     |
|                                                                        |
|  MAXIMUM ENTROPY RL:                                                   |
|  Maximizar: E[sum_t gamma^t * (r_t + alpha * H(pi(.|s_t)))]            |
|                                      ^^^^^^^^^^^^^^^^^^^^              |
|                                      Bonus de entropia                 |
|                                                                        |
|                                                                        |
|  POR QUE ENTROPIA?                                                     |
|  -----------------                                                     |
|                                                                        |
|  1. EXPLORACION: Politicas con alta entropia exploran mas              |
|                                                                        |
|  2. ROBUSTEZ: No colapsar a politica deterministica muy pronto         |
|                                                                        |
|  3. COMPOSICIONALIDAD: Politicas entrenadas se pueden combinar         |
|                                                                        |
|  4. MULTIMODALIDAD: Puede mantener multiples soluciones buenas         |
|                                                                        |
|                                                                        |
|  H(pi) = -E[log pi(a|s)]  = entropia de la politica                    |
|                                                                        |
|  Alta entropia: politica uniforme (maxima aleatoriedad)                |
|  Baja entropia: politica deterministica (poca aleatoriedad)            |
|                                                                        |
|  alpha (temperatura): controla balance reward vs entropia              |
|    - alpha alto: prioriza exploracion                                  |
|    - alpha bajo: prioriza explotacion                                  |
|    - SAC aprende alpha automaticamente!                                |
|                                                                        |
+------------------------------------------------------------------------+
```

### Arquitectura SAC

```
+------------------------------------------------------------------------+
|  ARQUITECTURA SAC                                                      |
+------------------------------------------------------------------------+
|                                                                        |
|  SAC usa TRES redes principales:                                       |
|                                                                        |
|                                                                        |
|  1. ACTOR (Policy Network):                                            |
|     -------------------------                                          |
|     pi(a|s; phi) = tanh(Normal(mu(s), sigma(s)))                       |
|                                                                        |
|     Para acciones continuas en [-1, 1]                                 |
|     (escalar al rango real del entorno)                                |
|                                                                        |
|     Estado s                                                           |
|        |                                                               |
|        v                                                               |
|     [Red Neuronal]                                                     |
|        |                                                               |
|        +---> mu(s)                                                     |
|        +---> log_sigma(s)                                              |
|                |                                                       |
|                v                                                       |
|     Sample: a = tanh(mu + sigma * epsilon)  donde epsilon ~ N(0,1)     |
|                                                                        |
|                                                                        |
|  2. CRITIC (Twin Q-Networks):                                          |
|     --------------------------                                         |
|     Q1(s, a; theta_1) y Q2(s, a; theta_2)                              |
|                                                                        |
|     Dos Q-networks para reducir sobreestimacion                        |
|     (como Double DQN pero para actor-critic)                           |
|                                                                        |
|     En el update se usa: min(Q1, Q2)                                   |
|                                                                        |
|                                                                        |
|  3. TARGET Q-Networks:                                                 |
|     --------------------                                               |
|     Q1'(theta_1') y Q2'(theta_2')                                      |
|                                                                        |
|     Actualizadas con soft update:                                      |
|     theta' <- tau * theta + (1 - tau) * theta'                         |
|     tau tipico = 0.005                                                 |
|                                                                        |
+------------------------------------------------------------------------+
```

### Algoritmo SAC

```
+------------------------------------------------------------------------+
|  ALGORITMO SAC                                                         |
+------------------------------------------------------------------------+
|                                                                        |
|  Inicializar:                                                          |
|    - Actor pi(a|s; phi)                                                |
|    - Critics Q1(s,a; theta_1), Q2(s,a; theta_2)                        |
|    - Target critics con mismos pesos                                   |
|    - Replay buffer D                                                   |
|    - Temperatura alpha (o aprender automaticamente)                    |
|                                                                        |
|  Para cada paso de entrenamiento:                                      |
|                                                                        |
|    1. Observar estado s, sample accion a ~ pi(.|s)                     |
|    2. Ejecutar a, observar r, s', done                                 |
|    3. Guardar (s, a, r, s', done) en D                                 |
|                                                                        |
|    4. Sample batch de D                                                |
|                                                                        |
|    5. Update Critics:                                                  |
|       a' ~ pi(.|s')                                                    |
|       target_Q = r + gamma * (1-done) * (min(Q1',Q2')(s',a') - alpha*log pi(a'|s'))
|       L_Q = MSE(Q1(s,a), target_Q) + MSE(Q2(s,a), target_Q)            |
|                                                                        |
|    6. Update Actor:                                                    |
|       a_new ~ pi(.|s)                                                  |
|       L_pi = E[alpha * log pi(a_new|s) - min(Q1,Q2)(s, a_new)]         |
|                                                                        |
|    7. Update Alpha (opcional, automatic tuning):                       |
|       L_alpha = E[-alpha * (log pi(a|s) + target_entropy)]             |
|                                                                        |
|    8. Soft update targets:                                             |
|       theta_i' <- tau * theta_i + (1-tau) * theta_i'                   |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion SAC

```python
"""
SAC (Soft Actor-Critic) implementacion completa.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from typing import Tuple, List
import matplotlib.pyplot as plt
from collections import deque
import random


class ReplayBuffer:
    """Buffer de experiencias."""

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """
    Politica gaussiana para acciones continuas.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna media y log_std."""
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample accion y calcula log_prob.

        Returns:
            (accion, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            # Reparameterization trick
            action = dist.rsample()

        # Aplicar tanh squashing
        action_squashed = torch.tanh(action)

        # Calcular log_prob con correccion por tanh
        log_prob = dist.log_prob(action)
        # Correccion: log_prob -= log(1 - tanh(a)^2)
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action_squashed, log_prob


class QNetwork(nn.Module):
    """Q-Network para SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TwinQNetwork(nn.Module):
    """Twin Q-Networks (Q1 y Q2)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)


class SACAgent:
    """
    Agente SAC completo.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Alpha (temperatura)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim  # Heuristica: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """Selecciona accion."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic)

        return action.cpu().numpy()[0]

    def update(self) -> dict:
        """Un paso de entrenamiento."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ===== Update Critic =====
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target_value = rewards + (1 - dones) * self.gamma * (
                q_target - self.alpha * next_log_probs
            )

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_value) + \
                      nn.functional.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Update Actor =====
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Update Alpha =====
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (
                log_probs + self.target_entropy
            ).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ===== Soft Update Targets =====
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0,
            "alpha": self.alpha
        }


def train_sac(
    env_name: str = "Pendulum-v1",
    n_episodes: int = 200,
    max_steps: int = 200,
    update_freq: int = 1,
    warmup_steps: int = 1000
) -> Tuple[SACAgent, List[float]]:
    """Entrena SAC."""
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space.high[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        auto_alpha=True
    )

    rewards_history = []
    total_steps = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            total_steps += 1

            # Warmup: acciones aleatorias
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action = action * action_scale  # Escalar al rango real

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Guardar en buffer (accion normalizada)
            agent.replay_buffer.push(
                state, action / action_scale, reward, next_state, done
            )

            # Update
            if total_steps >= warmup_steps and total_steps % update_freq == 0:
                metrics = agent.update()

            episode_reward += reward
            state = next_state

            if done:
                break

        rewards_history.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Alpha: {agent.alpha:.3f}")

    env.close()
    return agent, rewards_history


if __name__ == "__main__":
    print("=== Entrenando SAC en Pendulum ===\n")
    agent, rewards = train_sac(n_episodes=200)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('SAC - Curva de Aprendizaje')
    plt.grid(True, alpha=0.3)
    plt.savefig('sac_training.png', dpi=150)
    plt.show()
```

## 5. Continuous vs Discrete Action Spaces

```
+------------------------------------------------------------------------+
|  ACCIONES CONTINUAS vs DISCRETAS                                       |
+------------------------------------------------------------------------+
|                                                                        |
|  DISCRETAS:                                                            |
|  ----------                                                            |
|  - Conjunto finito: {izquierda, derecha, arriba, abajo}                |
|  - Representacion: Categorical distribution                            |
|  - Algoritmos: DQN, A2C, PPO con Categorical                           |
|                                                                        |
|  Salida de red:                                                        |
|    [logit_1, logit_2, ..., logit_n] --> Softmax --> Probabilidades     |
|                                                                        |
|                                                                        |
|  CONTINUAS:                                                            |
|  ----------                                                            |
|  - Valores reales: angulo en [-pi, pi], fuerza en [0, 100]             |
|  - Representacion: Gaussian/Normal distribution                        |
|  - Algoritmos: SAC, TD3, PPO con Normal                                |
|                                                                        |
|  Salida de red:                                                        |
|    mu(s), sigma(s) --> Normal(mu, sigma) --> Sample accion             |
|                                                                        |
|                                                                        |
|  HIBRIDAS (Parametrized Action Space):                                 |
|  -------------------------------------                                 |
|  - Accion discreta + parametros continuos                              |
|  - Ejemplo: {shoot, move(direccion), wait}                             |
|             shoot: sin parametros                                      |
|             move: direccion continua [0, 2pi]                          |
|                                                                        |
+------------------------------------------------------------------------+
```

### Codigo para ambos tipos

```python
"""
Actor-Critic para acciones discretas y continuas.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing import Tuple


class DiscreteActorCritic(nn.Module):
    """Actor-Critic para acciones discretas."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class ContinuousActorCritic(nn.Module):
    """Actor-Critic para acciones continuas."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        log_std_init: float = 0.0
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)

        # Log std como parametro aprendible (o red separada)
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        mean = self.actor_mean(features)
        value = self.critic(features)
        return mean, self.actor_log_std, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()  # Reparameterization trick

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1)
```

## 6. Aplicacion: Optimizacion de Firewall con SAC

```python
"""
Agente SAC para optimizacion dinamica de reglas de firewall.

El agente ajusta parametros continuos del firewall (rate limits,
timeouts, thresholds) para maximizar seguridad minimizando impacto
en trafico legitimo.
"""
import numpy as np
import torch
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class TrafficState:
    """Estado del trafico de red."""
    requests_per_second: float
    unique_ips: int
    avg_payload_size: float
    suspicious_ratio: float
    blocked_ratio: float
    latency_ms: float
    cpu_usage: float
    memory_usage: float


class FirewallEnvironment:
    """
    Entorno de firewall para RL.

    Acciones continuas:
    - rate_limit: [100, 10000] requests/sec
    - connection_timeout: [1, 60] segundos
    - block_threshold: [0.1, 0.9] ratio sospechoso para bloquear
    - whitelist_threshold: [0.01, 0.3] ratio para whitelist automatica
    """

    def __init__(self):
        self.state_dim = 8
        self.action_dim = 4

        self.action_low = np.array([100, 1, 0.1, 0.01])
        self.action_high = np.array([10000, 60, 0.9, 0.3])

        self.reset()

    def _get_state_vector(self) -> np.ndarray:
        """Convierte estado a vector normalizado."""
        return np.array([
            self.traffic.requests_per_second / 10000,
            self.traffic.unique_ips / 1000,
            self.traffic.avg_payload_size / 10000,
            self.traffic.suspicious_ratio,
            self.traffic.blocked_ratio,
            self.traffic.latency_ms / 1000,
            self.traffic.cpu_usage,
            self.traffic.memory_usage
        ], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reinicia entorno."""
        self.step_count = 0
        self.total_attacks_blocked = 0
        self.total_legitimate_blocked = 0
        self.attack_intensity = 0.2

        # Estado inicial
        self.traffic = TrafficState(
            requests_per_second=1000,
            unique_ips=100,
            avg_payload_size=500,
            suspicious_ratio=0.1,
            blocked_ratio=0.0,
            latency_ms=10,
            cpu_usage=0.3,
            memory_usage=0.4
        )

        return self._get_state_vector()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta accion (ajustes de firewall).

        action: valores en [-1, 1] que se escalan al rango real
        """
        self.step_count += 1

        # Escalar acciones al rango real
        action_scaled = (action + 1) / 2  # [0, 1]
        action_real = self.action_low + action_scaled * (self.action_high - self.action_low)

        rate_limit = action_real[0]
        timeout = action_real[1]
        block_threshold = action_real[2]
        whitelist_threshold = action_real[3]

        # Simular efecto de acciones
        # Generar trafico (con variacion de ataque)
        if self.step_count % 50 == 0:
            self.attack_intensity = np.random.uniform(0.1, 0.5)

        base_rps = np.random.uniform(800, 1200)
        attack_rps = base_rps * (1 + self.attack_intensity * 2)

        self.traffic.requests_per_second = attack_rps
        self.traffic.suspicious_ratio = self.attack_intensity + np.random.uniform(-0.05, 0.05)

        # Calcular bloqueos basado en configuracion
        traffic_over_limit = max(0, attack_rps - rate_limit) / attack_rps

        blocked_by_threshold = self.traffic.suspicious_ratio > block_threshold

        attack_traffic = self.attack_intensity * attack_rps
        legitimate_traffic = (1 - self.attack_intensity) * attack_rps

        # Ataques bloqueados
        if blocked_by_threshold:
            attacks_blocked = attack_traffic * 0.9  # 90% efectividad
            legitimate_blocked = legitimate_traffic * 0.1  # 10% falsos positivos
        else:
            attacks_blocked = attack_traffic * traffic_over_limit
            legitimate_blocked = legitimate_traffic * traffic_over_limit * 0.5

        self.total_attacks_blocked += attacks_blocked
        self.total_legitimate_blocked += legitimate_blocked

        # Actualizar metricas
        self.traffic.blocked_ratio = (attacks_blocked + legitimate_blocked) / attack_rps
        self.traffic.latency_ms = 10 + (rate_limit / 1000) * 5  # Rate limiting aumenta latencia
        self.traffic.cpu_usage = 0.3 + self.traffic.blocked_ratio * 0.3
        self.traffic.memory_usage = 0.4 + (timeout / 60) * 0.2

        # Calcular recompensa
        # + por ataques bloqueados
        # - por trafico legitimo bloqueado
        # - por latencia alta
        # - por recursos altos
        reward = (
            attacks_blocked / 100 * 10 -            # Ataques bloqueados: bueno
            legitimate_blocked / 100 * 50 -         # Legitimo bloqueado: muy malo
            self.traffic.latency_ms / 100 * 2 -     # Latencia: malo
            self.traffic.cpu_usage * 5 -            # CPU: malo
            self.traffic.memory_usage * 3           # Memoria: malo
        )

        # Bonus por buena deteccion
        if self.traffic.suspicious_ratio > 0.3 and blocked_by_threshold:
            reward += 10  # Bloqueo correcto bajo ataque

        done = self.step_count >= 500

        info = {
            "attacks_blocked": attacks_blocked,
            "legitimate_blocked": legitimate_blocked,
            "rate_limit": rate_limit,
            "block_threshold": block_threshold
        }

        return self._get_state_vector(), reward, done, info


def train_firewall_agent(n_episodes: int = 100) -> Tuple[SACAgent, List[float]]:
    """Entrena agente de firewall con SAC."""
    env = FirewallEnvironment()

    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        auto_alpha=True,
        batch_size=128
    )

    rewards_history = []
    metrics_history = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_attacks_blocked = 0
        episode_legitimate_blocked = 0

        for step in range(500):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

            episode_reward += reward
            episode_attacks_blocked += info["attacks_blocked"]
            episode_legitimate_blocked += info["legitimate_blocked"]

            state = next_state
            if done:
                break

        rewards_history.append(episode_reward)
        metrics_history.append({
            "attacks_blocked": episode_attacks_blocked,
            "legitimate_blocked": episode_legitimate_blocked,
            "false_positive_rate": episode_legitimate_blocked / (episode_attacks_blocked + episode_legitimate_blocked + 1)
        })

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_fpr = np.mean([m["false_positive_rate"] for m in metrics_history[-10:]])
            print(f"Episode {episode + 1} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg FPR: {avg_fpr:.3f} | "
                  f"Alpha: {agent.alpha:.3f}")

    return agent, rewards_history


if __name__ == "__main__":
    print("=== Entrenando Firewall Agent con SAC ===\n")
    agent, rewards = train_firewall_agent(n_episodes=100)

    # Graficar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(rewards, alpha=0.3)
    axes[0].plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), linewidth=2)
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_title('Curva de Aprendizaje')
    axes[0].grid(True, alpha=0.3)

    # Evaluar politica final
    env = FirewallEnvironment()
    state = env.reset()
    actions_log = []

    for _ in range(100):
        action = agent.select_action(state, deterministic=True)
        state, _, done, info = env.step(action)
        actions_log.append(info)
        if done:
            break

    rate_limits = [a["rate_limit"] for a in actions_log]
    thresholds = [a["block_threshold"] for a in actions_log]

    axes[1].plot(rate_limits, label='Rate Limit')
    axes[1].plot(np.array(thresholds) * 10000, label='Block Threshold (x10000)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Valor')
    axes[1].set_title('Acciones del Agente (Evaluacion)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('firewall_sac_training.png', dpi=150)
    plt.show()
```

## 7. Resumen

```
+------------------------------------------------------------------------+
|  ACTOR-CRITIC METHODS - RESUMEN                                        |
+------------------------------------------------------------------------+
|                                                                        |
|  ACTOR-CRITIC BASICO:                                                  |
|    - Actor: aprende politica pi(a|s)                                   |
|    - Critic: aprende funcion de valor V(s) o Q(s,a)                    |
|    - Combina ventajas de value-based y policy-based                    |
|                                                                        |
|  A2C (Advantage Actor-Critic):                                         |
|    - Usa Advantage A(s,a) = Q(s,a) - V(s)                              |
|    - Estimado con TD error: delta = r + gamma*V(s') - V(s)             |
|    - GAE para balance bias-varianza                                    |
|    - Entornos paralelos sincronizados                                  |
|                                                                        |
|  A3C (Asynchronous A2C):                                               |
|    - Workers asincronos                                                |
|    - Cada worker actualiza modelo global                               |
|    - Datos decorrelacionados naturalmente                              |
|    - En practica, A2C con GPU es igual de bueno                        |
|                                                                        |
|  SAC (Soft Actor-Critic):                                              |
|    - Maximum entropy RL                                                |
|    - Twin Q-networks (reduce sobreestimacion)                          |
|    - Alpha automatico (balance exploration-exploitation)               |
|    - Estado del arte para acciones continuas                           |
|    - Off-policy (sample efficient)                                     |
|                                                                        |
|  CUANDO USAR QUE:                                                      |
|  ----------------                                                      |
|  - Discreto, on-policy rapido: A2C                                     |
|  - Continuo, sample efficient: SAC                                     |
|  - Continuo, on-policy: PPO (ver capitulo anterior)                    |
|  - Distribuido masivo: A3C o PPO distribuido                           |
|                                                                        |
|  APLICACIONES EN SEGURIDAD:                                            |
|    - Optimizacion de firewalls                                         |
|    - Respuesta automatica a incidentes                                 |
|    - Ajuste dinamico de IDS                                            |
|    - Control de honeypots                                              |
|                                                                        |
+------------------------------------------------------------------------+
```

---

**Siguiente:** Multi-Agent Reinforcement Learning (MARL)
