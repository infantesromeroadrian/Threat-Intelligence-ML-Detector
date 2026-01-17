# Aplicaciones de RL en Ciberseguridad

## 1. Vision General

### El Potencial de RL en Seguridad

```
+------------------------------------------------------------------------+
|  POR QUE RL PARA CIBERSEGURIDAD?                                       |
+------------------------------------------------------------------------+
|                                                                        |
|  CARACTERISTICAS DE SEGURIDAD QUE ENCAJAN CON RL:                      |
|                                                                        |
|  1. ENTORNO DINAMICO                                                   |
|     - Atacantes evolucionan constantemente                             |
|     - Nuevas vulnerabilidades emergen                                  |
|     - Infraestructura cambia                                           |
|                                                                        |
|  2. SECUENCIAS DE DECISIONES                                           |
|     - Penetration testing: scan -> exploit -> pivot -> exfil           |
|     - Respuesta incidentes: detect -> contain -> eradicate -> recover  |
|                                                                        |
|  3. FEEDBACK RETRASADO                                                 |
|     - Exito de un ataque puede revelarse mucho despues                 |
|     - Efectividad de defensa solo se ve con ataques reales             |
|                                                                        |
|  4. TRADE-OFFS COMPLEJOS                                               |
|     - Seguridad vs usabilidad                                          |
|     - Falsos positivos vs falsos negativos                             |
|     - Coste de defensa vs riesgo de ataque                             |
|                                                                        |
|                                                                        |
|  APLICACIONES PRINCIPALES:                                             |
|  -------------------------                                             |
|                                                                        |
|  +-------------------+    +-------------------+    +-------------------+|
|  | OFFENSIVE         |    | DEFENSIVE         |    | ADAPTIVE          ||
|  | - Pentesting      |    | - IDS tuning      |    | - MTD             ||
|  | - Exploit dev     |    | - Response        |    | - Deception       ||
|  | - Red team        |    | - Network defense |    | - Honeypots       ||
|  +-------------------+    +-------------------+    +-------------------+|
|                                                                        |
+------------------------------------------------------------------------+
```

## 2. Automated Penetration Testing

### Arquitectura del Sistema

```
+------------------------------------------------------------------------+
|  AUTOMATED PENTESTING CON RL                                           |
+------------------------------------------------------------------------+
|                                                                        |
|  OBJETIVO: Agente que automatiza el proceso de penetration testing     |
|                                                                        |
|                                                                        |
|       +------------------+                                             |
|       | RL Agent         |                                             |
|       | (DQN/PPO)        |                                             |
|       +--------+---------+                                             |
|                |                                                       |
|                v                                                       |
|       +------------------+                                             |
|       | Action Interface |                                             |
|       | - nmap wrapper   |                                             |
|       | - metasploit API |                                             |
|       | - custom tools   |                                             |
|       +--------+---------+                                             |
|                |                                                       |
|                v                                                       |
|       +------------------+         +------------------+                |
|       | Target Network   |<------->| Environment      |                |
|       | (Real/Simulated) |         | State Tracker    |                |
|       +------------------+         +------------------+                |
|                                              |                         |
|                                              v                         |
|                                    +------------------+                |
|                                    | Reward Calculator|                |
|                                    | - Access gained  |                |
|                                    | - Stealth        |                |
|                                    | - Coverage       |                |
|                                    +------------------+                |
|                                                                        |
|                                                                        |
|  CICLO DE PENTESTING:                                                  |
|  --------------------                                                  |
|                                                                        |
|  Reconnaissance --> Scanning --> Gaining Access --> Maintaining Access |
|        |              |               |                   |            |
|        v              v               v                   v            |
|     [nmap]        [nikto]         [metasploit]       [persistence]     |
|     [whois]       [nessus]        [custom]           [C2]              |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion Completa

```python
"""
Automated Penetration Testing Agent usando RL.

NOTA: Este codigo es para propositos EDUCATIVOS.
Usar solo en entornos autorizados y con permiso explicito.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import random
import subprocess
import json


class PentestPhase(Enum):
    """Fases del penetration testing."""
    RECON = auto()
    SCANNING = auto()
    EXPLOITATION = auto()
    POST_EXPLOITATION = auto()
    EXFILTRATION = auto()


class PentestAction(Enum):
    """Acciones disponibles para el agente."""
    # Reconnaissance
    NMAP_DISCOVERY = 0
    NMAP_SERVICE_SCAN = 1
    NMAP_VULN_SCAN = 2

    # Exploitation
    EXPLOIT_SSH_BRUTE = 3
    EXPLOIT_WEB_SQLI = 4
    EXPLOIT_WEB_RCE = 5
    EXPLOIT_SMB_ETERNAL = 6

    # Post-exploitation
    PRIVESC_LINUX = 7
    PRIVESC_WINDOWS = 8
    CREDENTIAL_DUMP = 9

    # Movement
    LATERAL_SSH = 10
    LATERAL_SMB = 11
    LATERAL_PSEXEC = 12

    # Exfiltration
    EXFIL_HTTP = 13
    EXFIL_DNS = 14

    # Stealth
    CLEAR_LOGS = 15
    WAIT = 16


@dataclass
class Host:
    """Representa un host en la red."""
    ip: str
    hostname: str = ""
    os: str = "unknown"
    open_ports: Set[int] = field(default_factory=set)
    services: Dict[int, str] = field(default_factory=dict)
    vulnerabilities: List[str] = field(default_factory=list)
    credentials: List[Tuple[str, str]] = field(default_factory=list)
    access_level: int = 0  # 0=none, 1=user, 2=admin, 3=root/system
    discovered: bool = False
    scanned: bool = False


@dataclass
class PentestState:
    """Estado completo del pentest."""
    hosts: Dict[str, Host] = field(default_factory=dict)
    current_host: Optional[str] = None
    compromised_hosts: Set[str] = field(default_factory=set)
    collected_credentials: List[Tuple[str, str, str]] = field(default_factory=list)
    flags_captured: Set[str] = field(default_factory=set)
    detection_level: float = 0.0
    step_count: int = 0


class PentestEnvironment:
    """
    Entorno simulado de pentesting.

    Para produccion real, reemplazar metodos _execute_* con
    llamadas reales a herramientas (con MUCHO cuidado).
    """

    def __init__(
        self,
        target_network: str = "192.168.1.0/24",
        simulation_mode: bool = True
    ):
        self.target_network = target_network
        self.simulation_mode = simulation_mode
        self.max_steps = 500
        self.detection_threshold = 0.8

        # Definir red simulada
        self._setup_simulated_network()

        self.state = PentestState()
        self.reset()

    def _setup_simulated_network(self):
        """Configura red simulada para entrenamiento."""
        self.network_config = {
            "192.168.1.1": {
                "hostname": "router",
                "os": "RouterOS",
                "ports": {22: "ssh", 80: "http", 443: "https"},
                "vulns": [],
                "difficulty": "hard"
            },
            "192.168.1.10": {
                "hostname": "webserver",
                "os": "Linux",
                "ports": {22: "ssh", 80: "http", 443: "https", 3306: "mysql"},
                "vulns": ["CVE-2021-44228", "weak_ssh_password"],
                "flag": "FLAG{web_compromised}",
                "difficulty": "medium"
            },
            "192.168.1.20": {
                "hostname": "dc01",
                "os": "Windows Server 2019",
                "ports": {135: "rpc", 445: "smb", 389: "ldap", 3389: "rdp"},
                "vulns": ["zerologon"],
                "flag": "FLAG{domain_admin}",
                "difficulty": "hard"
            },
            "192.168.1.30": {
                "hostname": "fileserver",
                "os": "Windows Server 2016",
                "ports": {445: "smb", 3389: "rdp"},
                "vulns": ["ms17-010"],
                "flag": "FLAG{file_server}",
                "difficulty": "easy"
            },
            "192.168.1.100": {
                "hostname": "workstation1",
                "os": "Windows 10",
                "ports": {135: "rpc", 445: "smb"},
                "vulns": ["weak_local_admin"],
                "difficulty": "medium"
            },
            "192.168.1.200": {
                "hostname": "db01",
                "os": "Linux",
                "ports": {22: "ssh", 3306: "mysql", 5432: "postgres"},
                "vulns": ["mysql_root_no_password"],
                "flag": "FLAG{database_pwned}",
                "difficulty": "easy"
            }
        }

    @property
    def state_dim(self) -> int:
        """Dimension del vector de estado."""
        # Por cada host potencial: discovered, scanned, access_level, n_vulns, n_ports
        # + features globales
        max_hosts = 10
        per_host = 6
        global_features = 5
        return max_hosts * per_host + global_features

    @property
    def action_dim(self) -> int:
        """Numero de acciones."""
        return len(PentestAction)

    def _get_state_vector(self) -> np.ndarray:
        """Convierte estado a vector para el agente."""
        state_vec = []

        # Features por host
        sorted_hosts = sorted(self.state.hosts.keys())[:10]
        for ip in sorted_hosts:
            host = self.state.hosts[ip]
            state_vec.extend([
                1.0 if host.discovered else 0.0,
                1.0 if host.scanned else 0.0,
                host.access_level / 3.0,
                len(host.vulnerabilities) / 5.0,
                len(host.open_ports) / 10.0,
                1.0 if ip in self.state.compromised_hosts else 0.0
            ])

        # Padding si hay menos hosts
        while len(state_vec) < 10 * 6:
            state_vec.extend([0.0] * 6)

        # Features globales
        state_vec.extend([
            len(self.state.compromised_hosts) / 10.0,
            len(self.state.flags_captured) / 5.0,
            self.state.detection_level,
            len(self.state.collected_credentials) / 10.0,
            self.state.step_count / self.max_steps
        ])

        return np.array(state_vec, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reinicia el entorno."""
        self.state = PentestState()
        self.state.step_count = 0

        # Inicializar hosts desde config
        for ip, config in self.network_config.items():
            self.state.hosts[ip] = Host(
                ip=ip,
                hostname=config["hostname"],
                os=config["os"]
            )

        # El agente empieza sin conocer nada
        return self._get_state_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta una accion.

        Returns:
            (state, reward, done, info)
        """
        self.state.step_count += 1
        action_enum = PentestAction(action)

        reward = -0.01  # Pequeno coste por paso
        info = {"action": action_enum.name, "result": "none"}

        # Incrementar deteccion basado en accion
        detection_risk = self._get_detection_risk(action_enum)
        if random.random() < detection_risk:
            self.state.detection_level += random.uniform(0.05, 0.15)

        # Ejecutar accion
        if action_enum == PentestAction.NMAP_DISCOVERY:
            result = self._execute_discovery()
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum == PentestAction.NMAP_SERVICE_SCAN:
            result = self._execute_service_scan()
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum == PentestAction.NMAP_VULN_SCAN:
            result = self._execute_vuln_scan()
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum in [PentestAction.EXPLOIT_SSH_BRUTE,
                            PentestAction.EXPLOIT_WEB_SQLI,
                            PentestAction.EXPLOIT_WEB_RCE,
                            PentestAction.EXPLOIT_SMB_ETERNAL]:
            result = self._execute_exploit(action_enum)
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum in [PentestAction.PRIVESC_LINUX,
                            PentestAction.PRIVESC_WINDOWS]:
            result = self._execute_privesc(action_enum)
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum == PentestAction.CREDENTIAL_DUMP:
            result = self._execute_credential_dump()
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum in [PentestAction.LATERAL_SSH,
                            PentestAction.LATERAL_SMB,
                            PentestAction.LATERAL_PSEXEC]:
            result = self._execute_lateral_movement(action_enum)
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum in [PentestAction.EXFIL_HTTP,
                            PentestAction.EXFIL_DNS]:
            result = self._execute_exfiltration(action_enum)
            reward += result["reward"]
            info["result"] = result["message"]

        elif action_enum == PentestAction.CLEAR_LOGS:
            self.state.detection_level = max(0, self.state.detection_level - 0.1)
            reward += 0.5
            info["result"] = "logs_cleared"

        elif action_enum == PentestAction.WAIT:
            self.state.detection_level = max(0, self.state.detection_level - 0.05)
            info["result"] = "waiting"

        # Check condiciones de fin
        done = False

        if self.state.detection_level >= self.detection_threshold:
            reward -= 50  # Penalizacion por deteccion
            done = True
            info["result"] = "DETECTED"

        if self.state.step_count >= self.max_steps:
            done = True
            info["result"] = "MAX_STEPS"

        # Bonus por capturar todas las flags
        if len(self.state.flags_captured) >= 4:  # Todas las flags
            reward += 100
            done = True
            info["result"] = "ALL_FLAGS_CAPTURED"

        info["compromised"] = len(self.state.compromised_hosts)
        info["flags"] = len(self.state.flags_captured)
        info["detection"] = self.state.detection_level

        return self._get_state_vector(), reward, done, info

    def _get_detection_risk(self, action: PentestAction) -> float:
        """Retorna riesgo de deteccion de una accion."""
        risks = {
            PentestAction.NMAP_DISCOVERY: 0.05,
            PentestAction.NMAP_SERVICE_SCAN: 0.10,
            PentestAction.NMAP_VULN_SCAN: 0.20,
            PentestAction.EXPLOIT_SSH_BRUTE: 0.30,
            PentestAction.EXPLOIT_WEB_SQLI: 0.15,
            PentestAction.EXPLOIT_WEB_RCE: 0.20,
            PentestAction.EXPLOIT_SMB_ETERNAL: 0.40,
            PentestAction.PRIVESC_LINUX: 0.15,
            PentestAction.PRIVESC_WINDOWS: 0.20,
            PentestAction.CREDENTIAL_DUMP: 0.25,
            PentestAction.LATERAL_SSH: 0.10,
            PentestAction.LATERAL_SMB: 0.15,
            PentestAction.LATERAL_PSEXEC: 0.25,
            PentestAction.EXFIL_HTTP: 0.30,
            PentestAction.EXFIL_DNS: 0.10,
            PentestAction.CLEAR_LOGS: 0.05,
            PentestAction.WAIT: 0.0
        }
        return risks.get(action, 0.1)

    def _execute_discovery(self) -> Dict:
        """Ejecuta descubrimiento de hosts."""
        discovered = 0

        for ip, host in self.state.hosts.items():
            if not host.discovered and random.random() > 0.3:
                host.discovered = True
                discovered += 1

        return {
            "reward": discovered * 2.0,
            "message": f"discovered_{discovered}_hosts"
        }

    def _execute_service_scan(self) -> Dict:
        """Escanea servicios en hosts descubiertos."""
        scanned = 0

        for ip, host in self.state.hosts.items():
            if host.discovered and not host.scanned:
                config = self.network_config.get(ip, {})
                host.open_ports = set(config.get("ports", {}).keys())
                host.services = config.get("ports", {})
                host.os = config.get("os", "unknown")
                host.scanned = True
                scanned += 1

        return {
            "reward": scanned * 3.0,
            "message": f"scanned_{scanned}_hosts"
        }

    def _execute_vuln_scan(self) -> Dict:
        """Escanea vulnerabilidades."""
        vulns_found = 0

        for ip, host in self.state.hosts.items():
            if host.scanned and not host.vulnerabilities:
                config = self.network_config.get(ip, {})
                host.vulnerabilities = config.get("vulns", [])
                vulns_found += len(host.vulnerabilities)

        return {
            "reward": vulns_found * 5.0,
            "message": f"found_{vulns_found}_vulns"
        }

    def _execute_exploit(self, action: PentestAction) -> Dict:
        """Intenta explotar vulnerabilidad."""
        exploit_map = {
            PentestAction.EXPLOIT_SSH_BRUTE: ("weak_ssh_password", ["ssh"]),
            PentestAction.EXPLOIT_WEB_SQLI: ("sqli", ["http", "https"]),
            PentestAction.EXPLOIT_WEB_RCE: ("CVE-2021-44228", ["http", "https"]),
            PentestAction.EXPLOIT_SMB_ETERNAL: ("ms17-010", ["smb"])
        }

        vuln_needed, services_needed = exploit_map.get(action, (None, []))

        for ip, host in self.state.hosts.items():
            if ip in self.state.compromised_hosts:
                continue

            if not host.scanned:
                continue

            # Check si tiene la vulnerabilidad y servicio
            has_vuln = vuln_needed in host.vulnerabilities
            has_service = any(
                svc in host.services.values()
                for svc in services_needed
            )

            if has_vuln and has_service:
                # Exito!
                host.access_level = 1  # User level
                self.state.compromised_hosts.add(ip)
                self.state.current_host = ip

                # Capturar flag si existe
                config = self.network_config.get(ip, {})
                if "flag" in config:
                    self.state.flags_captured.add(config["flag"])
                    return {
                        "reward": 30.0,
                        "message": f"exploited_{ip}_FLAG_CAPTURED"
                    }

                return {
                    "reward": 15.0,
                    "message": f"exploited_{ip}"
                }

        return {
            "reward": -1.0,
            "message": "exploit_failed"
        }

    def _execute_privesc(self, action: PentestAction) -> Dict:
        """Intenta escalada de privilegios."""
        if not self.state.current_host:
            return {"reward": -1.0, "message": "no_current_host"}

        host = self.state.hosts.get(self.state.current_host)
        if not host:
            return {"reward": -1.0, "message": "invalid_host"}

        if host.access_level >= 3:
            return {"reward": 0.0, "message": "already_root"}

        # Probabilidad de exito basada en dificultad
        config = self.network_config.get(self.state.current_host, {})
        difficulty = config.get("difficulty", "medium")
        success_prob = {"easy": 0.8, "medium": 0.5, "hard": 0.2}.get(difficulty, 0.5)

        if random.random() < success_prob:
            host.access_level = 3
            return {
                "reward": 20.0,
                "message": f"privesc_success_{self.state.current_host}"
            }

        return {"reward": -2.0, "message": "privesc_failed"}

    def _execute_credential_dump(self) -> Dict:
        """Extrae credenciales del host actual."""
        if not self.state.current_host:
            return {"reward": -1.0, "message": "no_current_host"}

        host = self.state.hosts.get(self.state.current_host)
        if not host or host.access_level < 2:
            return {"reward": -1.0, "message": "insufficient_access"}

        # Simular extraccion de credenciales
        new_creds = []
        for other_ip in self.state.hosts:
            if random.random() > 0.6:
                new_creds.append((other_ip, "admin", "password123"))

        self.state.collected_credentials.extend(new_creds)

        return {
            "reward": len(new_creds) * 5.0,
            "message": f"dumped_{len(new_creds)}_creds"
        }

    def _execute_lateral_movement(self, action: PentestAction) -> Dict:
        """Movimiento lateral a otro host."""
        if not self.state.compromised_hosts:
            return {"reward": -1.0, "message": "no_compromised_hosts"}

        # Buscar targets validos
        targets = []
        for ip, host in self.state.hosts.items():
            if ip in self.state.compromised_hosts:
                continue
            if not host.discovered:
                continue

            # Check servicios necesarios
            if action == PentestAction.LATERAL_SSH and 22 in host.open_ports:
                targets.append(ip)
            elif action in [PentestAction.LATERAL_SMB, PentestAction.LATERAL_PSEXEC]:
                if 445 in host.open_ports:
                    targets.append(ip)

        if not targets:
            return {"reward": -0.5, "message": "no_valid_targets"}

        # Intentar movimiento con credenciales
        if self.state.collected_credentials:
            target_ip = random.choice(targets)
            host = self.state.hosts[target_ip]
            host.access_level = 1
            self.state.compromised_hosts.add(target_ip)
            self.state.current_host = target_ip

            # Check flag
            config = self.network_config.get(target_ip, {})
            if "flag" in config:
                self.state.flags_captured.add(config["flag"])
                return {
                    "reward": 25.0,
                    "message": f"lateral_{target_ip}_FLAG"
                }

            return {
                "reward": 10.0,
                "message": f"lateral_to_{target_ip}"
            }

        return {"reward": -1.0, "message": "no_credentials"}

    def _execute_exfiltration(self, action: PentestAction) -> Dict:
        """Exfiltra datos."""
        if not self.state.compromised_hosts:
            return {"reward": -1.0, "message": "no_access"}

        # Simular exfiltracion exitosa
        return {
            "reward": 5.0,
            "message": "data_exfiltrated"
        }


class PentestDQNAgent:
    """
    Agente DQN para pentesting automatizado.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100000,
        batch_size: int = 64
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, *args):
        self.buffer.append(args)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_pentest_agent(n_episodes: int = 1000):
    """Entrena agente de pentesting."""
    env = PentestEnvironment(simulation_mode=True)
    agent = PentestDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )

    rewards_history = []
    flags_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()

            total_reward += reward
            state = next_state

            if done:
                break

        if episode % 10 == 0:
            agent.update_target()

        agent.decay_epsilon()

        rewards_history.append(total_reward)
        flags_history.append(info.get("flags", 0))

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_flags = np.mean(flags_history[-100:])
            print(f"Episode {episode + 1} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Flags: {avg_flags:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, rewards_history
```

## 3. Intrusion Detection Adaptativo

### Sistema IDS con RL

```
+------------------------------------------------------------------------+
|  IDS ADAPTATIVO CON RL                                                 |
+------------------------------------------------------------------------+
|                                                                        |
|  PROBLEMA: IDS estaticos generan muchos falsos positivos               |
|            y fallan ante nuevos ataques                                |
|                                                                        |
|  SOLUCION: Agente RL que aprende a:                                    |
|    1. Ajustar umbrales dinamicamente                                   |
|    2. Seleccionar features relevantes                                  |
|    3. Balancear deteccion vs falsos positivos                          |
|                                                                        |
|                                                                        |
|                  Network Traffic                                       |
|                       |                                                |
|                       v                                                |
|                +-------------+                                         |
|                | Feature     |                                         |
|                | Extraction  |                                         |
|                +------+------+                                         |
|                       |                                                |
|                       v                                                |
|  +--------+    +-------------+    +---------+                          |
|  | ML     |    | RL Agent    |    | Action  |                          |
|  | Model  |--->| (Threshold  |--->| Execute |                          |
|  | Score  |    |  Selector)  |    |         |                          |
|  +--------+    +-------------+    +---------+                          |
|                       |                |                               |
|                       v                v                               |
|                +-------------+   [Alert/Block/Log]                     |
|                | Feedback    |                                         |
|                | (SOC/Auto)  |                                         |
|                +-------------+                                         |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion IDS RL

```python
"""
IDS Adaptativo usando Reinforcement Learning.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
from collections import deque


class IDSAction(Enum):
    """Acciones del IDS."""
    ALLOW = 0       # Permitir trafico
    LOG = 1         # Registrar para analisis
    ALERT = 2       # Generar alerta
    BLOCK = 3       # Bloquear trafico
    INVESTIGATE = 4 # Analisis profundo (costoso)


@dataclass
class NetworkFlow:
    """Flujo de red con features."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    duration: float
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    flags: int
    label: int  # 0=normal, 1=attack (para evaluacion)


class IDSEnvironment:
    """
    Entorno de IDS para RL.

    Simula flujo continuo de trafico con ataques.
    """

    def __init__(
        self,
        attack_ratio: float = 0.1,
        base_detection_rate: float = 0.7,
        fp_cost: float = -5.0,
        fn_cost: float = -50.0,
        tp_reward: float = 10.0,
        tn_reward: float = 0.1
    ):
        self.attack_ratio = attack_ratio
        self.base_detection_rate = base_detection_rate
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.tp_reward = tp_reward
        self.tn_reward = tn_reward

        # Features del estado
        self.window_size = 100
        self.state_dim = 15
        self.action_dim = len(IDSAction)

        # Historial de trafico
        self.traffic_history: deque = deque(maxlen=self.window_size)
        self.ml_scores: deque = deque(maxlen=self.window_size)

        # Metricas
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.reset()

    def _generate_flow(self) -> NetworkFlow:
        """Genera flujo de red (normal o ataque)."""
        is_attack = np.random.random() < self.attack_ratio

        if is_attack:
            # Patrones de ataque variados
            attack_type = np.random.choice(["scan", "dos", "brute", "exfil"])

            if attack_type == "scan":
                flow = NetworkFlow(
                    timestamp=0,
                    src_ip=f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}",
                    dst_ip="192.168.1.1",
                    src_port=np.random.randint(1024, 65535),
                    dst_port=np.random.randint(1, 1024),
                    protocol=6,
                    duration=0.01,
                    bytes_sent=60,
                    bytes_recv=0,
                    packets_sent=1,
                    packets_recv=0,
                    flags=2,  # SYN
                    label=1
                )
            elif attack_type == "dos":
                flow = NetworkFlow(
                    timestamp=0,
                    src_ip=f"10.0.0.{np.random.randint(1, 255)}",
                    dst_ip="192.168.1.100",
                    src_port=np.random.randint(1024, 65535),
                    dst_port=80,
                    protocol=6,
                    duration=0.001,
                    bytes_sent=np.random.randint(100, 1500),
                    bytes_recv=0,
                    packets_sent=np.random.randint(100, 1000),
                    packets_recv=0,
                    flags=2,
                    label=1
                )
            elif attack_type == "brute":
                flow = NetworkFlow(
                    timestamp=0,
                    src_ip=f"10.0.0.{np.random.randint(1, 255)}",
                    dst_ip="192.168.1.50",
                    src_port=np.random.randint(1024, 65535),
                    dst_port=22,
                    protocol=6,
                    duration=np.random.uniform(0.5, 2.0),
                    bytes_sent=np.random.randint(500, 2000),
                    bytes_recv=np.random.randint(100, 500),
                    packets_sent=np.random.randint(10, 50),
                    packets_recv=np.random.randint(5, 20),
                    flags=24,  # ACK+PSH
                    label=1
                )
            else:  # exfil
                flow = NetworkFlow(
                    timestamp=0,
                    src_ip="192.168.1.100",
                    dst_ip=f"8.8.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}",
                    src_port=np.random.randint(1024, 65535),
                    dst_port=443,
                    protocol=6,
                    duration=np.random.uniform(10, 300),
                    bytes_sent=np.random.randint(100000, 10000000),
                    bytes_recv=np.random.randint(1000, 10000),
                    packets_sent=np.random.randint(100, 10000),
                    packets_recv=np.random.randint(50, 500),
                    flags=24,
                    label=1
                )
        else:
            # Trafico normal
            flow = NetworkFlow(
                timestamp=0,
                src_ip=f"192.168.1.{np.random.randint(2, 254)}",
                dst_ip=f"10.0.0.{np.random.randint(1, 255)}",
                src_port=np.random.randint(1024, 65535),
                dst_port=np.random.choice([80, 443, 53, 25]),
                protocol=np.random.choice([6, 17]),
                duration=np.random.uniform(0.1, 60),
                bytes_sent=np.random.randint(100, 10000),
                bytes_recv=np.random.randint(100, 50000),
                packets_sent=np.random.randint(5, 100),
                packets_recv=np.random.randint(5, 200),
                flags=24,
                label=0
            )

        return flow

    def _flow_to_features(self, flow: NetworkFlow) -> np.ndarray:
        """Convierte flujo a features."""
        return np.array([
            np.log1p(flow.duration),
            np.log1p(flow.bytes_sent),
            np.log1p(flow.bytes_recv),
            np.log1p(flow.packets_sent),
            np.log1p(flow.packets_recv),
            flow.bytes_sent / (flow.packets_sent + 1),  # avg packet size
            flow.bytes_recv / (flow.packets_recv + 1),
            flow.dst_port / 65535,
            flow.protocol / 255,
            flow.flags / 63
        ], dtype=np.float32)

    def _get_ml_score(self, flow: NetworkFlow) -> float:
        """
        Simula score de modelo ML base.

        En produccion, seria un modelo real (Random Forest, etc.)
        """
        # Simular: ataques tienen mayor score, pero con ruido
        if flow.label == 1:
            return np.clip(
                self.base_detection_rate + np.random.normal(0, 0.2),
                0, 1
            )
        else:
            return np.clip(
                (1 - self.base_detection_rate) * 0.3 + np.random.normal(0, 0.15),
                0, 1
            )

    def _get_state(self) -> np.ndarray:
        """Construye vector de estado."""
        if len(self.traffic_history) < 10:
            return np.zeros(self.state_dim, dtype=np.float32)

        # Features agregadas de ventana reciente
        recent_flows = list(self.traffic_history)[-50:]
        recent_scores = list(self.ml_scores)[-50:]

        state = [
            # Estadisticas de scores ML
            np.mean(recent_scores),
            np.std(recent_scores),
            np.max(recent_scores),
            np.percentile(recent_scores, 90),

            # Estadisticas de trafico
            np.mean([f.bytes_sent for f in recent_flows]) / 10000,
            np.mean([f.packets_sent for f in recent_flows]) / 100,
            np.mean([f.duration for f in recent_flows]) / 60,

            # Diversidad de puertos/IPs
            len(set(f.dst_port for f in recent_flows)) / 50,
            len(set(f.src_ip for f in recent_flows)) / 50,

            # Metricas del IDS
            self.tp / (self.tp + self.fn + 1),  # Recall
            self.tp / (self.tp + self.fp + 1),  # Precision
            self.fp / (self.fp + self.tn + 1),  # FPR

            # Score del flujo actual
            recent_scores[-1] if recent_scores else 0,

            # Tendencia de scores
            (np.mean(recent_scores[-10:]) - np.mean(recent_scores[:10]))
            if len(recent_scores) >= 20 else 0,

            # Ratio de alertas recientes
            sum(1 for s in recent_scores if s > 0.5) / len(recent_scores)
        ]

        return np.array(state, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reinicia entorno."""
        self.traffic_history.clear()
        self.ml_scores.clear()
        self.tp = self.tn = self.fp = self.fn = 0

        # Generar historial inicial
        for _ in range(self.window_size):
            flow = self._generate_flow()
            self.traffic_history.append(flow)
            self.ml_scores.append(self._get_ml_score(flow))

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Procesa flujo actual con accion seleccionada.
        """
        action_enum = IDSAction(action)
        current_flow = self.traffic_history[-1]
        current_score = self.ml_scores[-1]
        is_attack = current_flow.label == 1

        reward = 0.0
        info = {"action": action_enum.name}

        # Calcular recompensa basada en accion y ground truth
        if action_enum == IDSAction.ALLOW:
            if is_attack:
                reward = self.fn_cost  # False negative
                self.fn += 1
                info["result"] = "FN"
            else:
                reward = self.tn_reward  # True negative
                self.tn += 1
                info["result"] = "TN"

        elif action_enum == IDSAction.LOG:
            # Neutral - solo registro
            reward = -0.1
            info["result"] = "LOGGED"

        elif action_enum == IDSAction.ALERT:
            if is_attack:
                reward = self.tp_reward * 0.7  # TP pero solo alerta
                self.tp += 1
                info["result"] = "TP_ALERT"
            else:
                reward = self.fp_cost * 0.5  # FP menos grave
                self.fp += 1
                info["result"] = "FP_ALERT"

        elif action_enum == IDSAction.BLOCK:
            if is_attack:
                reward = self.tp_reward  # True positive!
                self.tp += 1
                info["result"] = "TP_BLOCK"
            else:
                reward = self.fp_cost  # False positive
                self.fp += 1
                info["result"] = "FP_BLOCK"

        elif action_enum == IDSAction.INVESTIGATE:
            # Analisis profundo - alto coste pero certeza
            reward = -2.0  # Coste del analisis
            if is_attack:
                reward += self.tp_reward * 0.5
                self.tp += 1
                info["result"] = "TP_INVESTIGATED"
            else:
                self.tn += 1
                info["result"] = "TN_INVESTIGATED"

        # Generar nuevo flujo
        new_flow = self._generate_flow()
        self.traffic_history.append(new_flow)
        self.ml_scores.append(self._get_ml_score(new_flow))

        # Calcular metricas
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        info["precision"] = precision
        info["recall"] = recall
        info["f1"] = f1
        info["ml_score"] = current_score
        info["is_attack"] = is_attack

        done = False  # Continuo
        return self._get_state(), reward, done, info


def train_ids_agent(n_steps: int = 100000):
    """Entrena agente IDS."""
    env = IDSEnvironment(attack_ratio=0.15)

    # Usar DQN simple
    agent = nn.Sequential(
        nn.Linear(env.state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, env.action_dim)
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    buffer = deque(maxlen=50000)

    epsilon = 1.0
    epsilon_decay = 0.9999
    epsilon_min = 0.05

    metrics = {
        "rewards": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    state = env.reset()

    for step in range(n_steps):
        # Select action
        if np.random.random() < epsilon:
            action = np.random.randint(env.action_dim)
        else:
            with torch.no_grad():
                q = agent(torch.FloatTensor(state))
                action = q.argmax().item()

        next_state, reward, done, info = env.step(action)

        buffer.append((state, action, reward, next_state, done))

        # Train
        if len(buffer) >= 64:
            batch = random.sample(buffer, 64)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))

            current_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = agent(next_states).max(1)[0]
                target_q = rewards + 0.99 * next_q

            loss = nn.functional.mse_loss(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        state = next_state

        # Log
        metrics["rewards"].append(reward)
        metrics["precision"].append(info["precision"])
        metrics["recall"].append(info["recall"])
        metrics["f1"].append(info["f1"])

        if (step + 1) % 5000 == 0:
            avg_reward = np.mean(metrics["rewards"][-5000:])
            avg_f1 = np.mean(metrics["f1"][-5000:])
            print(f"Step {step + 1} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Avg F1: {avg_f1:.3f} | "
                  f"Epsilon: {epsilon:.3f}")

    return agent, metrics
```

## 4. Moving Target Defense (MTD)

```
+------------------------------------------------------------------------+
|  MOVING TARGET DEFENSE                                                 |
+------------------------------------------------------------------------+
|                                                                        |
|  CONCEPTO: Cambiar continuamente la superficie de ataque               |
|            para dificultar reconocimiento y explotacion                |
|                                                                        |
|                                                                        |
|  DIMENSIONES DE CAMBIO:                                                |
|  ----------------------                                                |
|                                                                        |
|  1. NETWORK:                                                           |
|     - IP shuffling                                                     |
|     - Port hopping                                                     |
|     - Network topology changes                                         |
|                                                                        |
|  2. PLATFORM:                                                          |
|     - OS rotation                                                      |
|     - Software diversity                                               |
|     - Configuration randomization                                      |
|                                                                        |
|  3. APPLICATION:                                                       |
|     - Address Space Layout Randomization (ASLR)                        |
|     - Code diversity                                                   |
|     - Data format variation                                            |
|                                                                        |
|                                                                        |
|  RL PARA MTD:                                                          |
|  ------------                                                          |
|                                                                        |
|  Estado:                                                               |
|    - Configuracion actual                                              |
|    - Actividad de atacantes detectada                                  |
|    - Impacto en usuarios legitimos                                     |
|                                                                        |
|  Accion:                                                               |
|    - Que dimension cambiar                                             |
|    - Cuando cambiar                                                    |
|    - Cuanto cambiar                                                    |
|                                                                        |
|  Recompensa:                                                           |
|    - + por ataques frustrados                                          |
|    - - por impacto en operaciones                                      |
|    - - por coste del cambio                                            |
|                                                                        |
+------------------------------------------------------------------------+
```

## 5. Honeypot Optimization

### Sistema de Honeypots Adaptativo

```python
"""
Optimizacion de Honeypots con RL.

El agente aprende a configurar honeypots para
maximizar captura de inteligencia sobre atacantes.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class HoneypotService(Enum):
    """Servicios que puede emular un honeypot."""
    SSH = 0
    HTTP = 1
    FTP = 2
    SMB = 3
    MYSQL = 4
    TELNET = 5
    RDP = 6
    SMTP = 7


class HoneypotInteraction(Enum):
    """Nivel de interaccion del honeypot."""
    LOW = 0      # Solo log de conexiones
    MEDIUM = 1   # Emulacion basica
    HIGH = 2     # Emulacion completa (riesgoso)


@dataclass
class AttackerProfile:
    """Perfil de atacante observado."""
    source_ip: str
    techniques: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    sophistication: float = 0.5  # 0-1


class HoneypotEnvironment:
    """
    Entorno para optimizacion de honeypots.
    """

    def __init__(self, n_honeypots: int = 5):
        self.n_honeypots = n_honeypots
        self.n_services = len(HoneypotService)
        self.n_interaction_levels = len(HoneypotInteraction)

        # Configuracion de honeypots: [service, interaction_level]
        self.configs: List[Tuple[int, int]] = []

        # Metricas
        self.intelligence_gathered = 0
        self.attackers_profiles: List[AttackerProfile] = []
        self.detected_compromises = 0

        self.step_count = 0
        self.max_steps = 1000

        self.reset()

    @property
    def state_dim(self) -> int:
        # Config actual + metricas + actividad reciente
        return self.n_honeypots * 2 + 10

    @property
    def action_dim(self) -> int:
        # Cambiar servicio o interaccion de cada honeypot
        return self.n_honeypots * (self.n_services + self.n_interaction_levels)

    def _get_state(self) -> np.ndarray:
        state = []

        # Configuraciones actuales
        for service, interaction in self.configs:
            state.append(service / self.n_services)
            state.append(interaction / self.n_interaction_levels)

        # Metricas globales
        state.extend([
            self.intelligence_gathered / 100,
            len(self.attackers_profiles) / 20,
            self.detected_compromises / 10,
            self.step_count / self.max_steps
        ])

        # Actividad reciente por servicio
        activity = [0] * self.n_services
        for profile in self.attackers_profiles[-10:]:
            for target in profile.targets:
                try:
                    svc = HoneypotService[target.upper()].value
                    activity[svc] += 1
                except KeyError:
                    pass

        state.extend([a / 10 for a in activity][:6])  # Top 6

        return np.array(state, dtype=np.float32)

    def reset(self) -> np.ndarray:
        # Configuracion inicial aleatoria
        self.configs = [
            (np.random.randint(self.n_services),
             np.random.randint(self.n_interaction_levels))
            for _ in range(self.n_honeypots)
        ]

        self.intelligence_gathered = 0
        self.attackers_profiles = []
        self.detected_compromises = 0
        self.step_count = 0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        reward = 0
        info = {}

        # Decodificar accion
        honeypot_idx = action // (self.n_services + self.n_interaction_levels)
        action_type = action % (self.n_services + self.n_interaction_levels)

        if honeypot_idx < self.n_honeypots:
            if action_type < self.n_services:
                # Cambiar servicio
                old_service = self.configs[honeypot_idx][0]
                self.configs[honeypot_idx] = (
                    action_type,
                    self.configs[honeypot_idx][1]
                )
                reward -= 0.5  # Coste de reconfiguracion
                info["change"] = f"hp{honeypot_idx}_service_{old_service}_to_{action_type}"
            else:
                # Cambiar nivel de interaccion
                new_level = action_type - self.n_services
                old_level = self.configs[honeypot_idx][1]
                self.configs[honeypot_idx] = (
                    self.configs[honeypot_idx][0],
                    new_level
                )
                reward -= 0.3
                info["change"] = f"hp{honeypot_idx}_level_{old_level}_to_{new_level}"

        # Simular actividad de atacantes
        if np.random.random() < 0.3:
            attacker = self._simulate_attacker()
            self.attackers_profiles.append(attacker)

            # Calcular inteligencia ganada
            for i, (service, interaction) in enumerate(self.configs):
                service_name = HoneypotService(service).name

                if service_name in attacker.targets:
                    # Honeypot atrajo al atacante!
                    base_intel = 5 * (interaction + 1)
                    intel = base_intel * attacker.sophistication

                    self.intelligence_gathered += intel
                    reward += intel

                    info[f"hp{i}_hit"] = service_name

                    # Riesgo de compromiso en high interaction
                    if interaction == 2 and np.random.random() < 0.1:
                        self.detected_compromises += 1
                        reward -= 20
                        info["compromise"] = i

        # Bonus por diversidad de servicios
        unique_services = len(set(c[0] for c in self.configs))
        reward += unique_services * 0.1

        done = self.step_count >= self.max_steps

        info["intelligence"] = self.intelligence_gathered
        info["profiles"] = len(self.attackers_profiles)

        return self._get_state(), reward, done, info

    def _simulate_attacker(self) -> AttackerProfile:
        """Simula comportamiento de atacante."""
        sophistication = np.random.beta(2, 5)  # Mas atacantes simples

        # Servicios que el atacante busca
        n_targets = np.random.randint(1, 4)
        targets = np.random.choice(
            [s.name for s in HoneypotService],
            n_targets,
            replace=False
        ).tolist()

        techniques = []
        if sophistication > 0.7:
            techniques = ["custom_exploit", "zero_day", "living_off_land"]
        elif sophistication > 0.4:
            techniques = ["metasploit", "nmap", "hydra"]
        else:
            techniques = ["script_kiddie", "default_creds"]

        return AttackerProfile(
            source_ip=f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}",
            techniques=techniques,
            targets=targets,
            sophistication=sophistication
        )


def train_honeypot_agent(n_episodes: int = 500):
    """Entrena agente para honeypots."""
    env = HoneypotEnvironment(n_honeypots=5)

    agent = nn.Sequential(
        nn.Linear(env.state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, env.action_dim)
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    rewards_history = []
    intel_history = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        epsilon = max(0.1, 1.0 - episode / 300)

        while True:
            if np.random.random() < epsilon:
                action = np.random.randint(env.action_dim)
            else:
                with torch.no_grad():
                    q = agent(torch.FloatTensor(state))
                    action = q.argmax().item()

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break

        rewards_history.append(episode_reward)
        intel_history.append(info.get("intelligence", 0))

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_intel = np.mean(intel_history[-50:])
            print(f"Episode {episode + 1} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Intel: {avg_intel:.2f}")

    return agent, rewards_history
```

## 6. Trading Algoritmico y Deteccion de Fraude

```
+------------------------------------------------------------------------+
|  RL EN FINANZAS Y FRAUDE                                               |
+------------------------------------------------------------------------+
|                                                                        |
|  TRADING ALGORITMICO:                                                  |
|  --------------------                                                  |
|                                                                        |
|  Estado:                                                               |
|    - Precios historicos                                                |
|    - Indicadores tecnicos                                              |
|    - Posicion actual                                                   |
|    - Capital disponible                                                |
|                                                                        |
|  Accion:                                                               |
|    - Buy / Sell / Hold                                                 |
|    - Cantidad                                                          |
|                                                                        |
|  Recompensa:                                                           |
|    - Profit/Loss                                                       |
|    - Risk-adjusted returns (Sharpe)                                    |
|                                                                        |
|                                                                        |
|  DETECCION DE FRAUDE:                                                  |
|  --------------------                                                  |
|                                                                        |
|  Estado:                                                               |
|    - Features de transaccion                                           |
|    - Historial de usuario                                              |
|    - Patrones de fraude conocidos                                      |
|                                                                        |
|  Accion:                                                               |
|    - Aprobar / Rechazar / Revisar                                      |
|    - Nivel de verificacion adicional                                   |
|                                                                        |
|  Recompensa:                                                           |
|    - + por fraude detectado                                            |
|    - - por falsos positivos (mal UX)                                   |
|    - -- por fraude no detectado                                        |
|                                                                        |
+------------------------------------------------------------------------+
```

## 7. Mejores Practicas

```
+------------------------------------------------------------------------+
|  MEJORES PRACTICAS PARA RL EN SEGURIDAD                                |
+------------------------------------------------------------------------+
|                                                                        |
|  1. SIMULACION ANTES DE PRODUCCION                                     |
|     ----------------------------------                                 |
|     - Entrenar en entornos simulados                                   |
|     - Validar exhaustivamente                                          |
|     - Desplegar gradualmente                                           |
|                                                                        |
|  2. SAFETY CONSTRAINTS                                                 |
|     ------------------                                                 |
|     - Limitar acciones peligrosas                                      |
|     - Implementar kill switches                                        |
|     - Human-in-the-loop para decisiones criticas                       |
|                                                                        |
|  3. REWARD ENGINEERING                                                 |
|     ------------------                                                 |
|     - Evitar reward hacking                                            |
|     - Considerar consecuencias a largo plazo                           |
|     - Penalizar comportamiento no deseado                              |
|                                                                        |
|  4. ROBUSTEZ                                                           |
|     ---------                                                          |
|     - Adversarial training                                             |
|     - Domain randomization                                             |
|     - Ensemble de politicas                                            |
|                                                                        |
|  5. EXPLICABILIDAD                                                     |
|     --------------                                                     |
|     - Logging detallado de decisiones                                  |
|     - Visualizacion de politica                                        |
|     - Post-hoc analysis                                                |
|                                                                        |
|  6. ACTUALIZACION CONTINUA                                             |
|     ----------------------                                             |
|     - Monitorear drift                                                 |
|     - Re-entrenar periodicamente                                       |
|     - A/B testing de politicas                                         |
|                                                                        |
+------------------------------------------------------------------------+
```

## 8. Herramientas y Frameworks

```
+------------------------------------------------------------------------+
|  HERRAMIENTAS PARA RL EN SEGURIDAD                                     |
+------------------------------------------------------------------------+
|                                                                        |
|  ENTORNOS DE SIMULACION:                                               |
|  -----------------------                                               |
|                                                                        |
|  - CyberBattleSim (Microsoft): Simulador de ataques                    |
|  - CALDERA (MITRE): Framework de adversary emulation                   |
|  - NetworkX: Simulacion de redes                                       |
|  - OpenAI Gym: Wrappers para entornos custom                           |
|                                                                        |
|                                                                        |
|  FRAMEWORKS DE RL:                                                     |
|  -----------------                                                     |
|                                                                        |
|  - Stable-Baselines3: Algoritmos robustos                              |
|  - RLlib (Ray): Distribuido y escalable                                |
|  - CleanRL: Implementaciones simples y educativas                      |
|  - TorchRL: Integracion nativa con PyTorch                             |
|                                                                        |
|                                                                        |
|  MONITORIZACION:                                                       |
|  ---------------                                                       |
|                                                                        |
|  - Weights & Biases: Tracking de experimentos                          |
|  - TensorBoard: Visualizacion                                          |
|  - MLflow: Lifecycle management                                        |
|                                                                        |
+------------------------------------------------------------------------+
```

## 9. Resumen

```
+------------------------------------------------------------------------+
|  APLICACIONES DE RL EN CIBERSEGURIDAD - RESUMEN                        |
+------------------------------------------------------------------------+
|                                                                        |
|  OFFENSIVE:                                                            |
|    - Automated pentesting                                              |
|    - Exploit discovery                                                 |
|    - Red team automation                                               |
|                                                                        |
|  DEFENSIVE:                                                            |
|    - IDS adaptativo                                                    |
|    - Respuesta automatica a incidentes                                 |
|    - Firewall dinamico                                                 |
|                                                                        |
|  ADAPTIVE:                                                             |
|    - Moving Target Defense                                             |
|    - Honeypot optimization                                             |
|    - Deception deployment                                              |
|                                                                        |
|  FINANCIAL:                                                            |
|    - Deteccion de fraude                                               |
|    - Trading algoritmico                                               |
|    - Anti-money laundering                                             |
|                                                                        |
|  CONSIDERACIONES:                                                      |
|    - Simular antes de produccion                                       |
|    - Implementar safety constraints                                    |
|    - Human-in-the-loop                                                 |
|    - Monitoreo continuo                                                |
|    - Explicabilidad                                                    |
|                                                                        |
|  FUTURO:                                                               |
|    - Agentes mas autonomos                                             |
|    - Multi-agent security                                              |
|    - Transfer learning entre dominios                                  |
|    - RL + LLM para decision making                                     |
|                                                                        |
+------------------------------------------------------------------------+
```

---

**Fin de la seccion de Reinforcement Learning**
