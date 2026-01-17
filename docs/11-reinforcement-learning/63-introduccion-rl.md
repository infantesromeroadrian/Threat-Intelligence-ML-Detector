# Introduccion a Reinforcement Learning

## 1. Que es Reinforcement Learning?

### El Paradigma del Aprendizaje por Refuerzo

```
+------------------------------------------------------------------------+
|  LOS TRES PARADIGMAS DE MACHINE LEARNING                               |
+------------------------------------------------------------------------+
|                                                                        |
|  SUPERVISADO              NO SUPERVISADO         REINFORCEMENT         |
|  -----------              --------------         LEARNING              |
|                                                  -------------         |
|  Datos: (X, y)            Datos: X               Datos: Interaccion    |
|  etiquetados              sin etiquetas          con entorno           |
|                                                                        |
|  Objetivo:                Objetivo:              Objetivo:             |
|  f(X) -> y                Encontrar              Maximizar             |
|                           patrones               recompensa            |
|                                                  acumulada             |
|                                                                        |
|  Ejemplo:                 Ejemplo:               Ejemplo:              |
|  Detectar spam            Segmentar              Robot que             |
|  (email -> spam/no)       clientes               aprende a andar       |
|                                                                        |
|  Feedback:                Feedback:              Feedback:             |
|  Etiqueta correcta        Ninguno                Recompensa/           |
|  inmediata                                       Penalizacion          |
|                                                  RETRASADA             |
|                                                                        |
+------------------------------------------------------------------------+
```

### Definicion Formal

**Reinforcement Learning (RL)** es un paradigma de aprendizaje donde un **agente** aprende a tomar **acciones** en un **entorno** para maximizar una **recompensa acumulada**.

```
+------------------------------------------------------------------------+
|  CICLO AGENTE-ENTORNO                                                  |
+------------------------------------------------------------------------+
|                                                                        |
|                         Estado (s_t)                                   |
|                    +------------------+                                |
|                    |                  |                                |
|                    v                  |                                |
|              +----------+        +----------+                          |
|              |          |        |          |                          |
|              |  AGENTE  |------->| ENTORNO  |                          |
|              |          | Accion |          |                          |
|              +----------+  (a_t) +----------+                          |
|                    ^                  |                                |
|                    |                  |                                |
|                    +------------------+                                |
|                     Recompensa (r_t)                                   |
|                     Nuevo Estado (s_{t+1})                             |
|                                                                        |
|  En cada paso t:                                                       |
|    1. Agente observa estado s_t                                        |
|    2. Agente elige accion a_t segun su politica                        |
|    3. Entorno devuelve recompensa r_t y nuevo estado s_{t+1}           |
|    4. Agente actualiza su conocimiento                                 |
|    5. Repetir hasta episodio termine                                   |
|                                                                        |
+------------------------------------------------------------------------+
```

## 2. Markov Decision Process (MDP)

### Definicion Matematica

Un MDP es una tupla **(S, A, P, R, gamma)** donde:

| Componente | Simbolo | Descripcion |
|------------|---------|-------------|
| **Estados** | S | Conjunto de todos los estados posibles |
| **Acciones** | A | Conjunto de todas las acciones posibles |
| **Transiciones** | P(s'|s,a) | Probabilidad de llegar a s' desde s tomando a |
| **Recompensas** | R(s,a,s') | Recompensa por transicion |
| **Descuento** | gamma | Factor de descuento [0,1] para recompensas futuras |

### Propiedad de Markov

```
+------------------------------------------------------------------------+
|  PROPIEDAD DE MARKOV: "El futuro solo depende del presente"            |
+------------------------------------------------------------------------+
|                                                                        |
|  P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0)                |
|                                                                        |
|                          =                                             |
|                                                                        |
|  P(s_{t+1} | s_t, a_t)                                                 |
|                                                                        |
|  El estado actual CAPTURA toda la informacion relevante del pasado     |
|                                                                        |
|  Ejemplo - Ajedrez:                                                    |
|    - Estado = posicion actual del tablero                              |
|    - No importa como llegamos a esa posicion                           |
|    - Todas las posibilidades futuras dependen solo de la posicion      |
|      actual                                                            |
|                                                                        |
+------------------------------------------------------------------------+
```

### Ejemplo: Robot en Grid World

```
+------------------------------------------------------------------------+
|  GRID WORLD - MDP SIMPLE                                               |
+------------------------------------------------------------------------+
|                                                                        |
|  +---+---+---+---+                                                     |
|  |   |   |   | G |   G = Goal (+10)                                    |
|  +---+---+---+---+   X = Trampa (-10)                                  |
|  |   | X |   |   |   R = Robot (agente)                                |
|  +---+---+---+---+   Movimientos: -1 cada paso                         |
|  | R |   |   |   |                                                     |
|  +---+---+---+---+                                                     |
|                                                                        |
|  Estados S: 12 celdas (posiciones)                                     |
|  Acciones A: {arriba, abajo, izquierda, derecha}                       |
|  Transiciones P: Deterministico (o estocastico con ruido)              |
|  Recompensas R:                                                        |
|    - Llegar a G: +10                                                   |
|    - Caer en X: -10                                                    |
|    - Cada paso: -1 (incentiva caminos cortos)                          |
|                                                                        |
+------------------------------------------------------------------------+
```

### Codigo: Definir un MDP Simple

```python
"""
MDP simple para Grid World.
Implementacion desde cero para entender los conceptos.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


class Action(Enum):
    """Acciones posibles del agente."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass(frozen=True)
class State:
    """Estado = posicion en el grid."""
    row: int
    col: int


class GridWorldMDP:
    """
    MDP para Grid World.

    Implementa la dinamica del entorno y las recompensas.
    """

    def __init__(
        self,
        rows: int = 3,
        cols: int = 4,
        goal_state: Tuple[int, int] = (0, 3),
        trap_state: Tuple[int, int] = (1, 1),
        step_reward: float = -0.1,
        goal_reward: float = 10.0,
        trap_reward: float = -10.0,
        gamma: float = 0.9
    ):
        self.rows = rows
        self.cols = cols
        self.goal = State(*goal_state)
        self.trap = State(*trap_state)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.gamma = gamma

        # Construir conjunto de estados
        self.states: List[State] = [
            State(r, c)
            for r in range(rows)
            for c in range(cols)
        ]

        # Estados terminales
        self.terminal_states = {self.goal, self.trap}

    def get_actions(self, state: State) -> List[Action]:
        """Retorna acciones disponibles en un estado."""
        if state in self.terminal_states:
            return []
        return list(Action)

    def transition(
        self,
        state: State,
        action: Action
    ) -> Tuple[State, float, bool]:
        """
        Ejecuta transicion.

        Returns:
            (nuevo_estado, recompensa, terminado)
        """
        if state in self.terminal_states:
            return state, 0.0, True

        # Calcular nueva posicion
        row, col = state.row, state.col

        if action == Action.UP:
            row = max(0, row - 1)
        elif action == Action.DOWN:
            row = min(self.rows - 1, row + 1)
        elif action == Action.LEFT:
            col = max(0, col - 1)
        elif action == Action.RIGHT:
            col = min(self.cols - 1, col + 1)

        new_state = State(row, col)

        # Calcular recompensa
        if new_state == self.goal:
            reward = self.goal_reward
            done = True
        elif new_state == self.trap:
            reward = self.trap_reward
            done = True
        else:
            reward = self.step_reward
            done = False

        return new_state, reward, done

    def get_transition_prob(
        self,
        state: State,
        action: Action,
        next_state: State
    ) -> float:
        """
        P(s'|s,a) - Probabilidad de transicion.

        En este MDP deterministico es 0 o 1.
        """
        actual_next, _, _ = self.transition(state, action)
        return 1.0 if actual_next == next_state else 0.0

    def render(self, agent_state: State | None = None) -> str:
        """Visualiza el grid."""
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                s = State(r, c)
                if agent_state and s == agent_state:
                    row.append(" R ")
                elif s == self.goal:
                    row.append(" G ")
                elif s == self.trap:
                    row.append(" X ")
                else:
                    row.append(" . ")
            grid.append("|" + "|".join(row) + "|")

        separator = "+" + "+".join(["---"] * self.cols) + "+"
        result = [separator]
        for row in grid:
            result.append(row)
            result.append(separator)

        return "\n".join(result)


# Demostrar el MDP
if __name__ == "__main__":
    mdp = GridWorldMDP()

    print("=== Grid World MDP ===")
    print(mdp.render(State(2, 0)))
    print(f"\nEstados: {len(mdp.states)}")
    print(f"Acciones: {[a.name for a in Action]}")
    print(f"Gamma: {mdp.gamma}")

    # Simular episodio aleatorio
    print("\n=== Episodio Aleatorio ===")
    state = State(2, 0)
    total_reward = 0
    step = 0

    while True:
        actions = mdp.get_actions(state)
        if not actions:
            break

        action = np.random.choice(actions)
        next_state, reward, done = mdp.transition(state, action)

        print(f"Paso {step}: {state} --{action.name}--> {next_state}, R={reward}")

        total_reward += reward
        state = next_state
        step += 1

        if done:
            break

    print(f"\nRecompensa total: {total_reward}")
```

## 3. Conceptos Fundamentales

### Estado, Accion, Recompensa

```
+------------------------------------------------------------------------+
|  COMPONENTES DEL RL                                                    |
+------------------------------------------------------------------------+
|                                                                        |
|  ESTADO (State, s)                                                     |
|  -----------------                                                     |
|  Descripcion del entorno en un momento dado.                           |
|                                                                        |
|  Ejemplos:                                                             |
|    - Juego: posicion del tablero                                       |
|    - Robot: posicion, velocidad, sensores                              |
|    - Trading: precios, volumen, indicadores                            |
|    - Seguridad: estado de la red, logs recientes                       |
|                                                                        |
|  ACCION (Action, a)                                                    |
|  ------------------                                                    |
|  Decision que toma el agente.                                          |
|                                                                        |
|  Tipos:                                                                |
|    - Discretas: {arriba, abajo, izquierda, derecha}                    |
|    - Continuas: angulo de direccion [-1, 1], aceleracion [0, 1]        |
|                                                                        |
|  RECOMPENSA (Reward, r)                                                |
|  ----------------------                                                |
|  Senal numerica de feedback.                                           |
|                                                                        |
|  Caracteristicas:                                                      |
|    - Puede ser positiva (premio) o negativa (castigo)                  |
|    - Puede ser densa (cada paso) o sparse (solo al final)              |
|    - Disenarla bien es CRUCIAL (reward shaping)                        |
|                                                                        |
+------------------------------------------------------------------------+
```

### Policy (Politica)

```
+------------------------------------------------------------------------+
|  POLITICA: pi(a|s) o pi(s)                                             |
+------------------------------------------------------------------------+
|                                                                        |
|  La politica es la "estrategia" del agente.                            |
|  Mapea estados a acciones.                                             |
|                                                                        |
|  POLITICA DETERMINISTICA:                                              |
|  ------------------------                                              |
|  pi: S -> A                                                            |
|  "En estado s, siempre hago accion a"                                  |
|                                                                        |
|  Ejemplo:                                                              |
|    pi(s1) = arriba                                                     |
|    pi(s2) = derecha                                                    |
|    pi(s3) = izquierda                                                  |
|                                                                        |
|  POLITICA ESTOCASTICA:                                                 |
|  ---------------------                                                 |
|  pi: S x A -> [0,1]                                                    |
|  pi(a|s) = P(accion=a | estado=s)                                      |
|  "En estado s, probabilidad de hacer a"                                |
|                                                                        |
|  Ejemplo:                                                              |
|    pi(arriba|s1) = 0.7                                                 |
|    pi(derecha|s1) = 0.2                                                |
|    pi(izquierda|s1) = 0.1                                              |
|                                                                        |
|  OBJETIVO: Encontrar politica optima pi* que maximice recompensa       |
|                                                                        |
+------------------------------------------------------------------------+
```

### Value Functions (Funciones de Valor)

```
+------------------------------------------------------------------------+
|  FUNCIONES DE VALOR                                                    |
+------------------------------------------------------------------------+
|                                                                        |
|  STATE-VALUE FUNCTION: V^pi(s)                                         |
|  -----------------------------                                         |
|  "Que tan bueno es estar en el estado s, siguiendo la politica pi"     |
|                                                                        |
|  V^pi(s) = E[R_t | s_t = s, pi]                                        |
|          = E[r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... | s_t = s]    |
|                                                                        |
|  donde R_t = suma de recompensas descontadas desde t                   |
|                                                                        |
|                                                                        |
|  ACTION-VALUE FUNCTION: Q^pi(s, a)                                     |
|  ---------------------------------                                     |
|  "Que tan bueno es tomar accion a en estado s, siguiendo pi despues"   |
|                                                                        |
|  Q^pi(s, a) = E[R_t | s_t = s, a_t = a, pi]                            |
|                                                                        |
|                                                                        |
|  RELACION:                                                             |
|  ---------                                                             |
|  V^pi(s) = sum_a pi(a|s) * Q^pi(s, a)                                  |
|                                                                        |
|  Para politica optima:                                                 |
|  V*(s) = max_a Q*(s, a)                                                |
|                                                                        |
+------------------------------------------------------------------------+
```

### Factor de Descuento (Gamma)

```
+------------------------------------------------------------------------+
|  FACTOR DE DESCUENTO: gamma (0 <= gamma <= 1)                          |
+------------------------------------------------------------------------+
|                                                                        |
|  Recompensa total descontada:                                          |
|                                                                        |
|  G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + gamma^3*r_{t+3} + ...   |
|      = sum_{k=0}^{inf} gamma^k * r_{t+k}                               |
|                                                                        |
|  gamma = 0:   Solo importa recompensa inmediata (miope)                |
|  gamma = 1:   Todas las recompensas importan igual (puede diverger)    |
|  gamma = 0.9: Balance tipico                                           |
|                                                                        |
|                                                                        |
|  Ejemplo con gamma = 0.9:                                              |
|  -------------------------                                             |
|                                                                        |
|    t=0    t=1    t=2    t=3    t=4                                     |
|    r=1    r=1    r=1    r=10   r=0                                     |
|                                                                        |
|  G_0 = 1 + 0.9*1 + 0.81*1 + 0.729*10 + ...                             |
|      = 1 + 0.9 + 0.81 + 7.29 + ...                                     |
|      = 10.0 (aproximadamente)                                          |
|                                                                        |
|  Recompensas lejanas valen menos -> incentiva eficiencia               |
|                                                                        |
+------------------------------------------------------------------------+
```

## 4. Exploration vs Exploitation

### El Dilema

```
+------------------------------------------------------------------------+
|  EXPLORATION vs EXPLOITATION                                           |
+------------------------------------------------------------------------+
|                                                                        |
|  EXPLOITATION:                           EXPLORATION:                  |
|  "Usar lo que ya se"                     "Probar cosas nuevas"         |
|                                                                        |
|  +----------------+                      +----------------+             |
|  | Ir al          |                      | Probar         |             |
|  | restaurante    |                      | restaurante    |             |
|  | favorito       |                      | nuevo          |             |
|  +----------------+                      +----------------+             |
|        |                                       |                       |
|        v                                       v                       |
|  Recompensa                              Recompensa                    |
|  predecible                              incierta                      |
|  (buena comida)                          (puede ser mejor              |
|                                           o peor)                      |
|                                                                        |
|                                                                        |
|  PROBLEMA:                                                             |
|  ---------                                                             |
|  - Solo exploitation: Nunca descubres mejores opciones                 |
|  - Solo exploration: Nunca aprovechas lo aprendido                     |
|                                                                        |
|  SOLUCION: Balance dinamico                                            |
|    - Explorar mas al principio (aprender)                              |
|    - Explotar mas al final (aplicar conocimiento)                      |
|                                                                        |
+------------------------------------------------------------------------+
```

### Estrategias de Exploracion

```
+------------------------------------------------------------------------+
|  ESTRATEGIAS DE EXPLORACION                                            |
+------------------------------------------------------------------------+
|                                                                        |
|  1. EPSILON-GREEDY                                                     |
|     ---------------                                                    |
|     Con probabilidad epsilon: accion aleatoria                         |
|     Con probabilidad 1-epsilon: mejor accion conocida                  |
|                                                                        |
|     epsilon = 0.1 -> 10% exploracion, 90% explotacion                  |
|                                                                        |
|     Tipico: epsilon decay                                              |
|       epsilon = max(0.01, epsilon * 0.995)                             |
|                                                                        |
|                                                                        |
|  2. SOFTMAX / BOLTZMANN                                                |
|     ------------------                                                 |
|     P(a|s) = exp(Q(s,a)/tau) / sum_a' exp(Q(s,a')/tau)                 |
|                                                                        |
|     tau = temperatura                                                  |
|       tau alto -> mas aleatorio                                        |
|       tau bajo -> mas greedy                                           |
|                                                                        |
|                                                                        |
|  3. UCB (Upper Confidence Bound)                                       |
|     ---------------------------                                        |
|     a = argmax_a [Q(s,a) + c * sqrt(ln(t)/N(s,a))]                     |
|                                                                        |
|     Bonus de exploracion para acciones poco probadas                   |
|                                                                        |
|                                                                        |
|  4. ENTROPIA EN POLITICA                                               |
|     -------------------                                                |
|     Anadir bonus de entropia a la funcion objetivo                     |
|     Incentiva politicas con mas variedad                               |
|                                                                        |
+------------------------------------------------------------------------+
```

### Codigo: Epsilon-Greedy

```python
"""
Implementacion de estrategias de exploracion.
"""
import numpy as np
from typing import List, Callable


class EpsilonGreedy:
    """
    Estrategia epsilon-greedy con decay.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(
        self,
        q_values: np.ndarray,
        available_actions: List[int] | None = None
    ) -> int:
        """
        Selecciona accion usando epsilon-greedy.

        Args:
            q_values: Q-values para cada accion
            available_actions: Lista de acciones validas (opcional)

        Returns:
            Indice de la accion seleccionada
        """
        if available_actions is None:
            available_actions = list(range(len(q_values)))

        if np.random.random() < self.epsilon:
            # Exploracion: accion aleatoria
            return np.random.choice(available_actions)
        else:
            # Explotacion: mejor accion
            q_available = q_values[available_actions]
            best_idx = np.argmax(q_available)
            return available_actions[best_idx]

    def decay(self) -> None:
        """Reduce epsilon."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )


class BoltzmannExploration:
    """
    Exploracion Softmax/Boltzmann.
    """

    def __init__(
        self,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        tau_decay: float = 0.995
    ):
        self.tau = tau_start
        self.tau_end = tau_end
        self.tau_decay = tau_decay

    def select_action(
        self,
        q_values: np.ndarray,
        available_actions: List[int] | None = None
    ) -> int:
        """
        Selecciona accion usando distribucion Boltzmann.
        """
        if available_actions is None:
            available_actions = list(range(len(q_values)))

        q_available = q_values[available_actions]

        # Softmax con temperatura
        exp_q = np.exp((q_available - q_available.max()) / self.tau)
        probs = exp_q / exp_q.sum()

        return np.random.choice(available_actions, p=probs)

    def decay(self) -> None:
        """Reduce temperatura."""
        self.tau = max(self.tau_end, self.tau * self.tau_decay)


class UCBExploration:
    """
    Upper Confidence Bound exploration.
    """

    def __init__(self, c: float = 2.0):
        self.c = c
        self.action_counts: dict = {}
        self.total_steps = 0

    def select_action(
        self,
        state: int,
        q_values: np.ndarray,
        available_actions: List[int] | None = None
    ) -> int:
        """
        Selecciona accion usando UCB.
        """
        if available_actions is None:
            available_actions = list(range(len(q_values)))

        self.total_steps += 1

        # Inicializar conteos si es necesario
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(len(q_values))

        # Calcular bonus UCB
        ucb_values = np.zeros(len(available_actions))

        for i, action in enumerate(available_actions):
            n = self.action_counts[state][action]

            if n == 0:
                # Accion no probada -> infinito
                ucb_values[i] = float('inf')
            else:
                # Q-value + bonus de exploracion
                bonus = self.c * np.sqrt(np.log(self.total_steps) / n)
                ucb_values[i] = q_values[action] + bonus

        best_idx = np.argmax(ucb_values)
        selected_action = available_actions[best_idx]

        # Actualizar conteo
        self.action_counts[state][selected_action] += 1

        return selected_action


# Demo
if __name__ == "__main__":
    # Q-values de ejemplo
    q_values = np.array([1.0, 2.0, 1.5, 0.5])

    print("=== Estrategias de Exploracion ===\n")
    print(f"Q-values: {q_values}")
    print(f"Mejor accion: {np.argmax(q_values)}\n")

    # Epsilon-greedy
    eps = EpsilonGreedy(epsilon_start=0.3)
    actions = [eps.select_action(q_values) for _ in range(1000)]
    print(f"Epsilon-greedy (eps=0.3):")
    print(f"  Distribucion: {np.bincount(actions, minlength=4) / 1000}")

    # Boltzmann
    boltz = BoltzmannExploration(tau_start=1.0)
    actions = [boltz.select_action(q_values) for _ in range(1000)]
    print(f"\nBoltzmann (tau=1.0):")
    print(f"  Distribucion: {np.bincount(actions, minlength=4) / 1000}")
```

## 5. Taxonomia de Algoritmos RL

### Clasificacion General

```
+------------------------------------------------------------------------+
|  TAXONOMIA DE ALGORITMOS RL                                            |
+------------------------------------------------------------------------+
|                                                                        |
|                          REINFORCEMENT LEARNING                        |
|                                   |                                    |
|                +------------------+------------------+                  |
|                |                                     |                  |
|          MODEL-BASED                           MODEL-FREE              |
|          (Aprende modelo                       (No aprende             |
|           del entorno)                          modelo)                |
|                |                                     |                  |
|        +-------+-------+               +-------------+-------------+   |
|        |               |               |                           |   |
|    Dyna-Q         MCTS/AlphaZero   VALUE-BASED              POLICY-BASED|
|    World Models                    (Aprende Q/V)            (Aprende pi)|
|                                        |                           |   |
|                                +-------+-------+           +-------+   |
|                                |               |           |       |   |
|                            Q-Learning      SARSA      REINFORCE   PPO  |
|                            DQN             TD(lambda)  A2C/A3C    TRPO |
|                            Double DQN                  SAC             |
|                            Dueling DQN                                 |
|                                                                        |
+------------------------------------------------------------------------+
```

### Model-Free vs Model-Based

```
+------------------------------------------------------------------------+
|  MODEL-FREE vs MODEL-BASED                                             |
+------------------------------------------------------------------------+
|                                                                        |
|  MODEL-FREE:                                                           |
|  -----------                                                           |
|  - NO aprende como funciona el entorno                                 |
|  - Aprende directamente la politica o funcion de valor                 |
|  - Necesita MUCHA interaccion con el entorno                           |
|  - Mas simple de implementar                                           |
|                                                                        |
|  Ejemplos: Q-Learning, DQN, PPO, SAC                                   |
|                                                                        |
|  +----------+     +----------+                                         |
|  |          |     |          |                                         |
|  |  Agente  |<--->| Entorno  |  Interaccion directa                    |
|  |          |     |  (real)  |                                         |
|  +----------+     +----------+                                         |
|                                                                        |
|                                                                        |
|  MODEL-BASED:                                                          |
|  ------------                                                          |
|  - Aprende modelo del entorno (P y R)                                  |
|  - Puede "simular" en su cabeza (planning)                             |
|  - Mas eficiente en muestras                                           |
|  - Mas complejo, errores del modelo se propagan                        |
|                                                                        |
|  Ejemplos: Dyna-Q, AlphaZero, World Models                             |
|                                                                        |
|  +----------+     +----------+     +----------+                        |
|  |          |     |  Modelo  |     |          |                        |
|  |  Agente  |<--->| Aprendido|<--->| Entorno  |                        |
|  |          |     |          |     |  (real)  |                        |
|  +----------+     +----------+     +----------+                        |
|                        |                                               |
|                        v                                               |
|                   Simulacion                                           |
|                   (planning)                                           |
|                                                                        |
+------------------------------------------------------------------------+
```

### Value-Based vs Policy-Based

```
+------------------------------------------------------------------------+
|  VALUE-BASED vs POLICY-BASED                                           |
+------------------------------------------------------------------------+
|                                                                        |
|  VALUE-BASED:                                                          |
|  ------------                                                          |
|  - Aprende funcion de valor: Q(s,a) o V(s)                             |
|  - Deriva politica de Q: pi(s) = argmax_a Q(s,a)                       |
|  - Mejor para espacios de accion DISCRETOS                             |
|  - Ejemplos: Q-Learning, DQN, SARSA                                    |
|                                                                        |
|          Q-table o Red Neuronal                                        |
|          +---+---+---+---+                                             |
|  Estado  | a1| a2| a3| a4|                                             |
|  +---+   +---+---+---+---+                                             |
|  | s1|-->|2.1|1.5|0.8|3.2| <- Q-values                                 |
|  +---+   +---+---+---+---+                                             |
|  | s2|-->|1.2|2.8|1.1|0.9|                                             |
|  +---+   +---+---+---+---+                                             |
|                                                                        |
|  Politica: tomar accion con mayor Q                                    |
|                                                                        |
|                                                                        |
|  POLICY-BASED:                                                         |
|  -------------                                                         |
|  - Aprende politica DIRECTAMENTE: pi(a|s)                              |
|  - Puede manejar acciones CONTINUAS                                    |
|  - Puede aprender politicas estocasticas                               |
|  - Ejemplos: REINFORCE, PPO, A2C                                       |
|                                                                        |
|          Red Neuronal (Policy Network)                                 |
|          +---------------+                                             |
|  Estado  |               |                                             |
|  +---+   |    pi(a|s)    |---> P(a1|s) = 0.1                           |
|  | s |-->|               |---> P(a2|s) = 0.7  <- Probabilidades        |
|  +---+   |               |---> P(a3|s) = 0.2                           |
|          +---------------+                                             |
|                                                                        |
|  Para acciones continuas:                                              |
|  mu(s), sigma(s) -> Normal(mu, sigma) -> accion continua               |
|                                                                        |
+------------------------------------------------------------------------+
```

### Actor-Critic (Hibrido)

```
+------------------------------------------------------------------------+
|  ACTOR-CRITIC: Lo mejor de ambos mundos                                |
+------------------------------------------------------------------------+
|                                                                        |
|  Combina VALUE-BASED y POLICY-BASED:                                   |
|                                                                        |
|  +-------------------+          +-------------------+                  |
|  |       ACTOR       |          |      CRITIC       |                  |
|  |   (Policy-based)  |          |   (Value-based)   |                  |
|  +-------------------+          +-------------------+                  |
|  |                   |          |                   |                  |
|  |  pi(a|s; theta)   |          |   V(s; w) o       |                  |
|  |                   |          |   Q(s,a; w)       |                  |
|  |  "Que accion      |          |                   |                  |
|  |   tomar"          |          |  "Que tan buena   |                  |
|  |                   |          |   es la accion"   |                  |
|  +-------------------+          +-------------------+                  |
|           |                              |                             |
|           +------------+  +--------------+                             |
|                        |  |                                            |
|                        v  v                                            |
|                  +-------------+                                       |
|                  |   Entorno   |                                       |
|                  +-------------+                                       |
|                                                                        |
|  Flujo:                                                                |
|  1. Actor propone accion segun pi(a|s)                                 |
|  2. Entorno devuelve recompensa y nuevo estado                         |
|  3. Critic evalua que tan buena fue la accion                          |
|  4. Actor se actualiza usando el feedback del Critic                   |
|  5. Critic se actualiza con la recompensa real                         |
|                                                                        |
|  Ventajas:                                                             |
|  - Varianza mas baja que REINFORCE puro                                |
|  - Funciona con acciones continuas                                     |
|  - Mas estable que value-only                                          |
|                                                                        |
|  Ejemplos: A2C, A3C, SAC, PPO                                          |
|                                                                        |
+------------------------------------------------------------------------+
```

## 6. On-Policy vs Off-Policy

```
+------------------------------------------------------------------------+
|  ON-POLICY vs OFF-POLICY                                               |
+------------------------------------------------------------------------+
|                                                                        |
|  ON-POLICY:                                                            |
|  ----------                                                            |
|  - Aprende de experiencias generadas por la politica ACTUAL            |
|  - Cada vez que actualiza, necesita nuevos datos                       |
|  - No puede reusar experiencias viejas                                 |
|                                                                        |
|  Ejemplos: SARSA, A2C, PPO (con restricciones)                         |
|                                                                        |
|  Politica actual                                                       |
|       |                                                                |
|       v                                                                |
|  [Generar datos] --> [Aprender] --> [Actualizar politica]              |
|       ^                                   |                            |
|       +-----------------------------------+                            |
|       (datos viejos se descartan)                                      |
|                                                                        |
|                                                                        |
|  OFF-POLICY:                                                           |
|  -----------                                                           |
|  - Puede aprender de experiencias generadas por OTRA politica          |
|  - Usa replay buffer para reusar experiencias                          |
|  - Mas eficiente en muestras                                           |
|                                                                        |
|  Ejemplos: Q-Learning, DQN, SAC                                        |
|                                                                        |
|  Politica de comportamiento --> [Generar datos] --> [Replay Buffer]    |
|       (puede ser antigua)                                |             |
|                                                          v             |
|  Politica objetivo <-------- [Aprender] <------ [Sample batch]         |
|       (la que queremos                                                 |
|        optimizar)                                                      |
|                                                                        |
+------------------------------------------------------------------------+
```

## 7. Bellman Equations

### Ecuaciones Fundamentales

```
+------------------------------------------------------------------------+
|  ECUACIONES DE BELLMAN                                                 |
+------------------------------------------------------------------------+
|                                                                        |
|  BASE de casi todos los algoritmos de RL.                              |
|  Expresan el valor de un estado en terminos del siguiente.             |
|                                                                        |
|                                                                        |
|  BELLMAN EXPECTATION EQUATION (para politica fija pi):                 |
|  ----------------------------------------------------                  |
|                                                                        |
|  V^pi(s) = sum_a pi(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma*V^pi(s')]|
|                                                                        |
|  Q^pi(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * sum_a' pi(a'|s') * Q^pi(s',a')]|
|                                                                        |
|                                                                        |
|  BELLMAN OPTIMALITY EQUATION (para politica optima):                   |
|  --------------------------------------------------                    |
|                                                                        |
|  V*(s) = max_a sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]         |
|                                                                        |
|  Q*(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q*(s',a')]   |
|                                                                        |
|                                                                        |
|  INTUICION:                                                            |
|  ----------                                                            |
|  Valor de estar aqui = Recompensa inmediata + Valor descontado de     |
|                        donde puedo llegar                              |
|                                                                        |
|       V(s) = r + gamma * V(s')                                         |
|              |         |                                               |
|       inmediata    futuro descontado                                   |
|                                                                        |
+------------------------------------------------------------------------+
```

### Codigo: Value Iteration

```python
"""
Value Iteration: Resolver MDP usando ecuaciones de Bellman.
"""
import numpy as np
from typing import Dict, Tuple, List


def value_iteration(
    states: List[int],
    actions: List[int],
    transition_probs: Dict[Tuple[int, int, int], float],  # P(s'|s,a)
    rewards: Dict[Tuple[int, int, int], float],  # R(s,a,s')
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration para encontrar V* y politica optima.

    Args:
        states: Lista de estados
        actions: Lista de acciones
        transition_probs: Probabilidades de transicion
        rewards: Recompensas
        gamma: Factor de descuento
        theta: Umbral de convergencia
        max_iterations: Maximo de iteraciones

    Returns:
        V: Funcion de valor optima
        policy: Politica optima
    """
    n_states = len(states)
    n_actions = len(actions)

    # Inicializar V arbitrariamente
    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        delta = 0

        for s in states:
            v_old = V[s]

            # Bellman optimality update
            action_values = []
            for a in actions:
                value = 0
                for s_prime in states:
                    prob = transition_probs.get((s, a, s_prime), 0)
                    reward = rewards.get((s, a, s_prime), 0)
                    value += prob * (reward + gamma * V[s_prime])
                action_values.append(value)

            # V(s) = max_a Q(s,a)
            V[s] = max(action_values) if action_values else 0

            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            print(f"Value Iteration convergio en {iteration + 1} iteraciones")
            break

    # Extraer politica optima
    policy = np.zeros(n_states, dtype=int)

    for s in states:
        action_values = []
        for a in actions:
            value = 0
            for s_prime in states:
                prob = transition_probs.get((s, a, s_prime), 0)
                reward = rewards.get((s, a, s_prime), 0)
                value += prob * (reward + gamma * V[s_prime])
            action_values.append(value)

        policy[s] = np.argmax(action_values) if action_values else 0

    return V, policy


# Ejemplo: Grid World simple
if __name__ == "__main__":
    # Grid 2x2:
    # +---+---+
    # | 0 | 1 | <- Goal (estado 1)
    # +---+---+
    # | 2 | 3 |
    # +---+---+

    states = [0, 1, 2, 3]
    actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT

    # Transiciones deterministicas simplificadas
    # (estado, accion, estado_siguiente) -> probabilidad
    transitions = {
        # Estado 0
        (0, 0, 0): 1.0,  # UP -> queda en 0
        (0, 1, 2): 1.0,  # DOWN -> va a 2
        (0, 2, 0): 1.0,  # LEFT -> queda en 0
        (0, 3, 1): 1.0,  # RIGHT -> va a 1 (GOAL)
        # Estado 1 (terminal)
        (1, 0, 1): 1.0, (1, 1, 1): 1.0, (1, 2, 1): 1.0, (1, 3, 1): 1.0,
        # Estado 2
        (2, 0, 0): 1.0,  # UP -> va a 0
        (2, 1, 2): 1.0,  # DOWN -> queda en 2
        (2, 2, 2): 1.0,  # LEFT -> queda en 2
        (2, 3, 3): 1.0,  # RIGHT -> va a 3
        # Estado 3
        (3, 0, 1): 1.0,  # UP -> va a 1 (GOAL)
        (3, 1, 3): 1.0,  # DOWN -> queda en 3
        (3, 2, 2): 1.0,  # LEFT -> va a 2
        (3, 3, 3): 1.0,  # RIGHT -> queda en 3
    }

    # Recompensas
    rewards = {}
    for key in transitions:
        s, a, s_prime = key
        if s_prime == 1 and s != 1:  # Llegar al goal
            rewards[key] = 10.0
        elif s == 1:  # Ya en goal, no reward
            rewards[key] = 0.0
        else:
            rewards[key] = -0.1  # Pequeño coste por paso

    # Resolver
    V, policy = value_iteration(states, actions, transitions, rewards)

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    print("\n=== Resultados ===")
    print("\nFuncion de Valor V*:")
    print(f"  V(0)={V[0]:.2f}  V(1)={V[1]:.2f}")
    print(f"  V(2)={V[2]:.2f}  V(3)={V[3]:.2f}")

    print("\nPolitica Optima:")
    print(f"  pi(0)={action_names[policy[0]]}  pi(1)=TERMINAL")
    print(f"  pi(2)={action_names[policy[2]]}  pi(3)={action_names[policy[3]]}")
```

## 8. Aplicaciones en Ciberseguridad

```
+------------------------------------------------------------------------+
|  RL EN CIBERSEGURIDAD                                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  1. AUTOMATED PENETRATION TESTING                                      |
|     --------------------------------                                   |
|     Agente RL explora red objetivo                                     |
|     - Estado: hosts descubiertos, vulnerabilidades, accesos            |
|     - Accion: escanear, explotar, pivotar, exfiltrar                   |
|     - Recompensa: +10 root access, +5 user access, -1 detectado        |
|                                                                        |
|  2. INTRUSION DETECTION SYSTEMS (IDS)                                  |
|     -----------------------------------                                |
|     Agente decide cuando alertar                                       |
|     - Estado: metricas de red, logs, patrones                          |
|     - Accion: alertar, ignorar, investigar                             |
|     - Recompensa: +1 true positive, -10 false positive, -100 miss      |
|                                                                        |
|  3. ADAPTIVE DEFENSE                                                   |
|     -----------------                                                  |
|     Honeypots que aprenden a atraer atacantes                          |
|     - Estado: actividad del atacante, servicios expuestos              |
|     - Accion: cambiar servicios, modificar respuestas                  |
|     - Recompensa: +1 por segundo de engagement, +10 tecnica nueva      |
|                                                                        |
|  4. MALWARE ANALYSIS                                                   |
|     -----------------                                                  |
|     Agente navega malware para observar comportamiento                 |
|     - Estado: estado del sandbox, calls observadas                     |
|     - Accion: interacciones de usuario, inputs                         |
|     - Recompensa: +1 nuevo comportamiento descubierto                  |
|                                                                        |
|  5. NETWORK DEFENSE / FIREWALL                                         |
|     ---------------------------                                        |
|     Ajustar reglas dinamicamente                                       |
|     - Estado: trafico actual, alertas, recursos                        |
|     - Accion: bloquear IP, rate limit, permitir                        |
|     - Recompensa: -1 ataque exitoso, +0.1 trafico legitimo             |
|                                                                        |
+------------------------------------------------------------------------+
```

### Ejemplo: Agente de Red Team Simple

```python
"""
Agente RL simple para pentesting simulado.

NOTA: Esto es un ejemplo educativo.
En entornos reales se necesita autorizacion explicita.
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Set


class PentestAction(Enum):
    """Acciones del agente pentester."""
    SCAN_PORTS = 0
    SCAN_VULNS = 1
    EXPLOIT_SSH = 2
    EXPLOIT_WEB = 3
    PRIVILEGE_ESCALATION = 4
    LATERAL_MOVEMENT = 5


@dataclass
class Host:
    """Representa un host en la red."""
    ip: str
    ports_open: Set[int]
    vulnerabilities: Set[str]
    access_level: int  # 0=none, 1=user, 2=root
    discovered: bool = False


class PentestEnvironment:
    """
    Entorno simulado para pentesting.
    """

    def __init__(self, num_hosts: int = 5):
        self.num_hosts = num_hosts
        self.hosts: List[Host] = []
        self.current_host_idx = 0
        self.detected = False
        self.detection_probability = 0.1

        self._setup_network()

    def _setup_network(self) -> None:
        """Configura red simulada."""
        for i in range(self.num_hosts):
            ports = set(np.random.choice(
                [22, 80, 443, 3306, 5432, 8080],
                size=np.random.randint(1, 4),
                replace=False
            ))

            vulns = set()
            if 22 in ports and np.random.random() > 0.5:
                vulns.add("SSH_WEAK_CREDS")
            if 80 in ports and np.random.random() > 0.3:
                vulns.add("WEB_SQL_INJECTION")
            if np.random.random() > 0.7:
                vulns.add("KERNEL_EXPLOIT")

            self.hosts.append(Host(
                ip=f"192.168.1.{10 + i}",
                ports_open=ports,
                vulnerabilities=vulns,
                access_level=0,
                discovered=(i == 0)  # Primer host siempre descubierto
            ))

    def reset(self) -> np.ndarray:
        """Reinicia el entorno."""
        self._setup_network()
        self.current_host_idx = 0
        self.detected = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Retorna estado como vector.
        [puertos_conocidos, vulns_conocidas, access_levels, detected]
        """
        state = []

        for host in self.hosts:
            if host.discovered:
                state.extend([
                    len(host.ports_open) / 6,  # Normalizado
                    len(host.vulnerabilities) / 3,
                    host.access_level / 2
                ])
            else:
                state.extend([0, 0, 0])

        state.append(1 if self.detected else 0)

        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta accion.

        Returns:
            (nuevo_estado, recompensa, terminado, info)
        """
        action = PentestAction(action)
        reward = 0
        done = False
        info = {"action": action.name}

        current_host = self.hosts[self.current_host_idx]

        # Check deteccion
        if np.random.random() < self.detection_probability:
            self.detected = True
            reward = -50
            done = True
            info["result"] = "DETECTED"
            return self._get_state(), reward, done, info

        # Ejecutar accion
        if action == PentestAction.SCAN_PORTS:
            # Descubrir puertos de hosts adyacentes
            for host in self.hosts:
                if not host.discovered and np.random.random() > 0.5:
                    host.discovered = True
                    reward += 1
                    info["result"] = f"Discovered {host.ip}"

        elif action == PentestAction.SCAN_VULNS:
            # Ya tenemos las vulns, solo reward por info
            if current_host.discovered and current_host.vulnerabilities:
                reward += 0.5
                info["result"] = f"Found vulns: {current_host.vulnerabilities}"

        elif action == PentestAction.EXPLOIT_SSH:
            if "SSH_WEAK_CREDS" in current_host.vulnerabilities:
                if current_host.access_level == 0:
                    current_host.access_level = 1
                    reward += 10
                    info["result"] = "USER ACCESS GAINED"
            else:
                reward -= 1
                info["result"] = "Exploit failed"

        elif action == PentestAction.EXPLOIT_WEB:
            if "WEB_SQL_INJECTION" in current_host.vulnerabilities:
                if current_host.access_level == 0:
                    current_host.access_level = 1
                    reward += 10
                    info["result"] = "USER ACCESS GAINED"
            else:
                reward -= 1
                info["result"] = "Exploit failed"

        elif action == PentestAction.PRIVILEGE_ESCALATION:
            if (current_host.access_level == 1 and
                "KERNEL_EXPLOIT" in current_host.vulnerabilities):
                current_host.access_level = 2
                reward += 20
                info["result"] = "ROOT ACCESS GAINED"
            else:
                reward -= 2
                info["result"] = "Privesc failed"

        elif action == PentestAction.LATERAL_MOVEMENT:
            # Moverse a otro host
            discovered_hosts = [
                i for i, h in enumerate(self.hosts)
                if h.discovered and i != self.current_host_idx
            ]
            if discovered_hosts:
                self.current_host_idx = np.random.choice(discovered_hosts)
                reward += 0.5
                info["result"] = f"Moved to {self.hosts[self.current_host_idx].ip}"
            else:
                reward -= 0.5
                info["result"] = "No hosts to move to"

        # Check objetivo completado (root en todos)
        if all(h.access_level == 2 for h in self.hosts):
            reward += 100
            done = True
            info["result"] = "OBJECTIVE COMPLETE"

        return self._get_state(), reward, done, info

    @property
    def state_size(self) -> int:
        """Tamano del vector de estado."""
        return self.num_hosts * 3 + 1

    @property
    def action_size(self) -> int:
        """Numero de acciones."""
        return len(PentestAction)


# Demo
if __name__ == "__main__":
    env = PentestEnvironment(num_hosts=3)

    print("=== Pentest Environment Demo ===")
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")

    state = env.reset()
    print(f"\nInitial state: {state}")

    # Episodio aleatorio
    total_reward = 0
    for step in range(20):
        action = np.random.randint(env.action_size)
        next_state, reward, done, info = env.step(action)

        print(f"Step {step}: {info['action']} -> {info.get('result', '')} (R={reward})")

        total_reward += reward
        state = next_state

        if done:
            break

    print(f"\nTotal reward: {total_reward}")
```

## 9. Resumen

```
+------------------------------------------------------------------------+
|  REINFORCEMENT LEARNING - RESUMEN                                      |
+------------------------------------------------------------------------+
|                                                                        |
|  DEFINICION:                                                           |
|    Agente aprende a tomar acciones para maximizar recompensa           |
|    acumulada a traves de interaccion con el entorno.                   |
|                                                                        |
|  COMPONENTES:                                                          |
|    - Estado (s): Descripcion del entorno                               |
|    - Accion (a): Decision del agente                                   |
|    - Recompensa (r): Feedback numerico                                 |
|    - Politica (pi): Estrategia del agente                              |
|    - Valor (V, Q): Que tan bueno es un estado/accion                   |
|                                                                        |
|  MDP:                                                                  |
|    Tupla (S, A, P, R, gamma) que formaliza el problema                 |
|    Propiedad de Markov: futuro solo depende del presente               |
|                                                                        |
|  EXPLORACION vs EXPLOTACION:                                           |
|    - Exploracion: Probar acciones nuevas                               |
|    - Explotacion: Usar conocimiento actual                             |
|    - Estrategias: epsilon-greedy, softmax, UCB                         |
|                                                                        |
|  TAXONOMIA:                                                            |
|    - Model-free vs Model-based                                         |
|    - Value-based vs Policy-based vs Actor-Critic                       |
|    - On-policy vs Off-policy                                           |
|                                                                        |
|  ECUACIONES DE BELLMAN:                                                |
|    V(s) = E[r + gamma * V(s')]                                         |
|    Q(s,a) = E[r + gamma * max Q(s',a')]                                |
|                                                                        |
|  APLICACIONES EN SEGURIDAD:                                            |
|    - Penetration testing automatizado                                  |
|    - IDS adaptativo                                                    |
|    - Defensa de red dinamica                                           |
|    - Analisis de malware                                               |
|                                                                        |
+------------------------------------------------------------------------+
```

---

**Siguiente:** Q-Learning y Deep Q-Networks (DQN)
