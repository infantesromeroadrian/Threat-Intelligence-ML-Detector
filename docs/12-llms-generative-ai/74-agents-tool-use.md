# 74. LLM Agents y Tool Use

## Tabla de Contenidos

1. [Introduccion a LLM Agents](#introduccion)
2. [Arquitectura de Agentes](#arquitectura)
3. [Function Calling](#function-calling)
4. [Patron ReAct](#react)
5. [Planning y Decomposition](#planning)
6. [Memory: Short y Long Term](#memory)
7. [Frameworks: LangChain y LlamaIndex](#frameworks)
8. [Multi-Agent Systems](#multi-agent)
9. [Aplicaciones en Ciberseguridad](#ciberseguridad)

---

## 1. Introduccion a LLM Agents {#introduccion}

Los agentes LLM combinan razonamiento con capacidad de ejecutar acciones en el mundo.

### Que es un Agente?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUE ES UN LLM AGENT?                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LLM SIMPLE:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Input (prompt) ──► LLM ──► Output (texto)                      │       │
│  │                                                                  │       │
│  │  Solo genera texto, no puede actuar en el mundo                 │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  LLM AGENT:                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │              ┌─────────┐                                        │       │
│  │              │   LLM   │ ◄─── "Cerebro" del agente              │       │
│  │              │ (Brain) │                                        │       │
│  │              └────┬────┘                                        │       │
│  │                   │                                              │       │
│  │         ┌─────────┼─────────┐                                   │       │
│  │         │         │         │                                   │       │
│  │         ▼         ▼         ▼                                   │       │
│  │    ┌────────┐┌────────┐┌────────┐                              │       │
│  │    │ Tool 1 ││ Tool 2 ││ Tool 3 │ ◄─── Capacidades             │       │
│  │    │Search  ││ Code   ││Database│                              │       │
│  │    └────────┘└────────┘└────────┘                              │       │
│  │         │         │         │                                   │       │
│  │         └─────────┼─────────┘                                   │       │
│  │                   ▼                                              │       │
│  │            ┌───────────┐                                        │       │
│  │            │Environment│ ◄─── Mundo real o sistemas            │       │
│  │            └───────────┘                                        │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  COMPONENTES CLAVE:                                                        │
│  - LLM: Razonamiento y toma de decisiones                                  │
│  - Tools: Acciones que el agente puede ejecutar                           │
│  - Memory: Contexto de interacciones pasadas                              │
│  - Planning: Descomposicion de tareas complejas                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Arquitectura de Agentes {#arquitectura}

### Loop del Agente

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT LOOP                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌──────────────────┐                                │
│                        │   User Request   │                                │
│                        └────────┬─────────┘                                │
│                                 │                                           │
│                                 ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                        AGENT LOOP                                 │      │
│  │  ┌────────────────────────────────────────────────────────────┐  │      │
│  │  │                                                             │  │      │
│  │  │   1. PERCEIVE                                              │  │      │
│  │  │      └─► Analizar input + contexto + memoria               │  │      │
│  │  │                        │                                    │  │      │
│  │  │                        ▼                                    │  │      │
│  │  │   2. THINK                                                 │  │      │
│  │  │      └─► LLM razona sobre que hacer                        │  │      │
│  │  │          - Necesito informacion? → Tool                    │  │      │
│  │  │          - Puedo responder? → Output                       │  │      │
│  │  │          - Tarea compleja? → Plan                          │  │      │
│  │  │                        │                                    │  │      │
│  │  │              ┌─────────┴─────────┐                         │  │      │
│  │  │              ▼                   ▼                         │  │      │
│  │  │   3. ACT                    4. RESPOND                     │  │      │
│  │  │      └─► Ejecutar tool          └─► Generar respuesta      │  │      │
│  │  │          │                                                  │  │      │
│  │  │          ▼                                                  │  │      │
│  │  │   5. OBSERVE                                               │  │      │
│  │  │      └─► Procesar resultado del tool                       │  │      │
│  │  │          │                                                  │  │      │
│  │  │          └─────────► Back to THINK ──────┘                 │  │      │
│  │  │                                                             │  │      │
│  │  └────────────────────────────────────────────────────────────┘  │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                 │                                           │
│                                 ▼                                           │
│                        ┌──────────────────┐                                │
│                        │  Final Response  │                                │
│                        └──────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion Base

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class ActionType(Enum):
    TOOL_CALL = "tool_call"
    RESPOND = "respond"
    PLAN = "plan"


@dataclass
class Tool:
    """Herramienta que el agente puede usar."""
    name: str
    description: str
    parameters: dict[str, dict]  # JSON Schema
    function: Callable[..., Any]

    def execute(self, **kwargs) -> str:
        result = self.function(**kwargs)
        return str(result)


@dataclass
class AgentAction:
    """Accion decidida por el agente."""
    type: ActionType
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    response: Optional[str] = None
    reasoning: str = ""


@dataclass
class AgentState:
    """Estado del agente durante ejecucion."""
    messages: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    iterations: int = 0
    max_iterations: int = 10


class Agent:
    """Agente LLM base."""

    def __init__(
        self,
        llm,
        tools: list[Tool],
        system_prompt: str = "",
        max_iterations: int = 10
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_iterations = max_iterations

    def _default_system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        )
        return f"""You are an AI assistant with access to tools.

Available tools:
{tools_desc}

To use a tool, respond with:
TOOL: tool_name
ARGS: {{"param1": "value1", "param2": "value2"}}

To provide a final answer, respond with:
ANSWER: your response here

Always think step by step before acting."""

    def _parse_action(self, response: str) -> AgentAction:
        """Parsea la respuesta del LLM en una accion."""
        response = response.strip()

        # Check for tool call
        if "TOOL:" in response:
            lines = response.split("\n")
            tool_name = None
            tool_args = {}
            reasoning = ""

            for line in lines:
                if line.startswith("TOOL:"):
                    tool_name = line.replace("TOOL:", "").strip()
                elif line.startswith("ARGS:"):
                    import json
                    args_str = line.replace("ARGS:", "").strip()
                    tool_args = json.loads(args_str)
                else:
                    reasoning += line + "\n"

            return AgentAction(
                type=ActionType.TOOL_CALL,
                tool_name=tool_name,
                tool_args=tool_args,
                reasoning=reasoning.strip()
            )

        # Check for final answer
        if "ANSWER:" in response:
            answer = response.split("ANSWER:")[-1].strip()
            return AgentAction(
                type=ActionType.RESPOND,
                response=answer
            )

        # Default: treat as reasoning/continue
        return AgentAction(
            type=ActionType.RESPOND,
            response=response
        )

    def _execute_tool(self, action: AgentAction) -> str:
        """Ejecuta un tool y retorna el resultado."""
        if action.tool_name not in self.tools:
            return f"Error: Tool '{action.tool_name}' not found"

        tool = self.tools[action.tool_name]
        try:
            result = tool.execute(**(action.tool_args or {}))
            return result
        except Exception as e:
            return f"Error executing {action.tool_name}: {str(e)}"

    def _build_prompt(self, state: AgentState, user_query: str) -> str:
        """Construye el prompt con historial."""
        parts = [self.system_prompt, "", f"User Query: {user_query}", ""]

        for i, result in enumerate(state.tool_results):
            parts.append(f"Tool Call {i+1}: {result['tool']}")
            parts.append(f"Result: {result['result']}")
            parts.append("")

        parts.append("What should I do next?")
        return "\n".join(parts)

    def run(self, query: str) -> str:
        """Ejecuta el agente hasta obtener respuesta final."""
        state = AgentState(max_iterations=self.max_iterations)

        while state.iterations < state.max_iterations:
            state.iterations += 1

            # Build prompt
            prompt = self._build_prompt(state, query)

            # Get LLM response
            response = self.llm.generate(prompt)

            # Parse action
            action = self._parse_action(response)

            # Execute action
            if action.type == ActionType.RESPOND:
                return action.response

            elif action.type == ActionType.TOOL_CALL:
                result = self._execute_tool(action)
                state.tool_results.append({
                    "tool": action.tool_name,
                    "args": action.tool_args,
                    "result": result
                })

        return "Max iterations reached without final answer"
```

---

## 3. Function Calling {#function-calling}

Function calling es el mecanismo estandar de OpenAI para tool use.

```python
from typing import Callable, Any
import json


class FunctionCallingAgent:
    """Agente usando OpenAI function calling."""

    def __init__(self, client, model: str = "gpt-4"):
        self.client = client  # OpenAI client
        self.model = model
        self.functions: dict[str, Callable] = {}
        self.function_schemas: list[dict] = []

    def add_function(
        self,
        func: Callable,
        description: str,
        parameters: dict
    ) -> None:
        """Registra una funcion."""
        name = func.__name__
        self.functions[name] = func
        self.function_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })

    def run(self, messages: list[dict]) -> str:
        """Ejecuta el agente con function calling."""
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.function_schemas,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # Check if model wants to call a function
            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    # Execute function
                    if func_name in self.functions:
                        result = self.functions[func_name](**func_args)
                    else:
                        result = f"Function {func_name} not found"

                    # Add result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                # No function call, return response
                return message.content


# Ejemplo de uso
def search_vulnerabilities(cve_id: str) -> dict:
    """Busca informacion de una vulnerabilidad."""
    # Simulado - en produccion usar API real
    return {
        "cve_id": cve_id,
        "description": "SQL Injection vulnerability",
        "cvss": 8.5,
        "affected": ["product-1.0", "product-1.1"]
    }


def scan_target(target: str, scan_type: str = "quick") -> dict:
    """Escanea un objetivo."""
    return {
        "target": target,
        "status": "completed",
        "findings": ["Open port 22", "Open port 80"]
    }


# Setup agent
from openai import OpenAI
client = OpenAI()
agent = FunctionCallingAgent(client)

agent.add_function(
    search_vulnerabilities,
    "Search for information about a CVE vulnerability",
    {
        "type": "object",
        "properties": {
            "cve_id": {
                "type": "string",
                "description": "The CVE identifier (e.g., CVE-2024-1234)"
            }
        },
        "required": ["cve_id"]
    }
)

agent.add_function(
    scan_target,
    "Scan a target for security issues",
    {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "IP address or hostname to scan"
            },
            "scan_type": {
                "type": "string",
                "enum": ["quick", "full", "stealth"],
                "description": "Type of scan to perform"
            }
        },
        "required": ["target"]
    }
)
```

---

## 4. Patron ReAct {#react}

ReAct (Reason + Act) alterna entre razonamiento y acciones.

```python
class ReActAgent:
    """Agente usando el patron ReAct."""

    REACT_PROMPT = """Answer the following question using the available tools.
Use this exact format:

Thought: [your reasoning about what to do]
Action: [tool_name]
Action Input: [input for the tool]
Observation: [result from the tool]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: [your answer]

Available tools:
{tools}

Question: {question}

Begin!
"""

    def __init__(self, llm, tools: list[Tool], max_iterations: int = 10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations

    def _format_tools(self) -> str:
        return "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        )

    def run(self, question: str) -> str:
        """Ejecuta ReAct loop."""
        prompt = self.REACT_PROMPT.format(
            tools=self._format_tools(),
            question=question
        )

        scratchpad = ""

        for i in range(self.max_iterations):
            # Generate next step
            full_prompt = prompt + scratchpad
            response = self.llm.generate(full_prompt, stop=["Observation:"])

            scratchpad += response

            # Check for final answer
            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()

            # Parse and execute action
            if "Action:" in response and "Action Input:" in response:
                action_line = [l for l in response.split("\n") if l.startswith("Action:")][0]
                input_line = [l for l in response.split("\n") if l.startswith("Action Input:")][0]

                tool_name = action_line.replace("Action:", "").strip()
                tool_input = input_line.replace("Action Input:", "").strip()

                # Execute tool
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name].execute(input=tool_input)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                else:
                    result = f"Tool '{tool_name}' not found"

                scratchpad += f"\nObservation: {result}\n"

        return "Max iterations reached"
```

---

## 5. Planning y Decomposition {#planning}

### Estrategias de Planning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESTRATEGIAS DE PLANNING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PLAN-AND-EXECUTE                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Task ──► Planner ──► [Step 1, Step 2, Step 3, ...]            │       │
│  │                              │                                   │       │
│  │                    ┌─────────┼─────────┐                        │       │
│  │                    ▼         ▼         ▼                        │       │
│  │                Execute   Execute   Execute                      │       │
│  │                    │         │         │                        │       │
│  │                    └─────────┼─────────┘                        │       │
│  │                              ▼                                   │       │
│  │                         Aggregate                               │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  2. HIERARCHICAL TASK DECOMPOSITION                                        │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                     [Main Task]                                 │       │
│  │                    /     |     \                                │       │
│  │               [Sub1]  [Sub2]  [Sub3]                           │       │
│  │              /    \     |      /   \                           │       │
│  │           [a]   [b]   [c]    [d]   [e]                        │       │
│  │                                                                  │       │
│  │  Cada nivel puede replanificar si encuentra problemas          │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  3. LEAST-TO-MOST                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Descompone en subproblemas, resuelve desde el mas simple      │       │
│  │                                                                  │       │
│  │  Complex Problem                                                │       │
│  │       │                                                          │       │
│  │       ▼                                                          │       │
│  │  [Simple] ──► [Medium] ──► [Hard] ──► [Final]                  │       │
│  │     ▲            ▲           ▲                                  │       │
│  │     │            │           │                                  │       │
│  │   Solve     Use result   Use results                           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion

```python
@dataclass
class Plan:
    """Plan de ejecucion."""
    goal: str
    steps: list[str]
    current_step: int = 0
    results: list[str] = field(default_factory=list)


class PlanningAgent:
    """Agente con capacidad de planning."""

    def __init__(self, llm, tools: list[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.executor = ReActAgent(llm, tools)

    def create_plan(self, task: str) -> Plan:
        """Crea un plan para la tarea."""
        prompt = f"""Create a step-by-step plan to accomplish this task.
Each step should be specific and actionable.

Task: {task}

Available tools:
{chr(10).join(f"- {name}: {tool.description}" for name, tool in self.tools.items())}

Respond with numbered steps:
1. [First step]
2. [Second step]
...

Plan:"""

        response = self.llm.generate(prompt)

        # Parse steps
        steps = []
        for line in response.strip().split("\n"):
            if line and line[0].isdigit():
                step = line.split(".", 1)[-1].strip()
                if step:
                    steps.append(step)

        return Plan(goal=task, steps=steps)

    def execute_plan(self, plan: Plan) -> str:
        """Ejecuta el plan paso a paso."""
        for i, step in enumerate(plan.steps):
            plan.current_step = i

            # Ejecutar paso
            result = self.executor.run(
                f"Complete this step: {step}\n\nContext from previous steps: {plan.results}"
            )
            plan.results.append(f"Step {i+1}: {result}")

        # Sintetizar respuesta final
        synthesis_prompt = f"""Based on these results, provide the final answer to: {plan.goal}

Results:
{chr(10).join(plan.results)}

Final Answer:"""

        return self.llm.generate(synthesis_prompt)

    def run(self, task: str) -> str:
        """Planifica y ejecuta."""
        plan = self.create_plan(task)
        return self.execute_plan(plan)
```

---

## 6. Memory: Short y Long Term {#memory}

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import numpy as np


class Memory(ABC):
    """Interfaz base para memoria."""

    @abstractmethod
    def add(self, content: str, metadata: dict = None) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[dict]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ConversationMemory(Memory):
    """Memoria de conversacion (short-term)."""

    def __init__(self, max_messages: int = 50):
        self.messages: list[dict] = []
        self.max_messages = max_messages

    def add(self, content: str, metadata: dict = None) -> None:
        self.messages.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

        # Trim if too long
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def search(self, query: str, k: int = 5) -> list[dict]:
        # For conversation, just return recent messages
        return self.messages[-k:]

    def clear(self) -> None:
        self.messages = []

    def get_context(self, max_tokens: int = 2000) -> str:
        """Obtiene contexto de conversacion."""
        context = []
        total_chars = 0

        for msg in reversed(self.messages):
            msg_len = len(msg["content"])
            if total_chars + msg_len > max_tokens * 4:  # Approx chars per token
                break
            context.insert(0, msg["content"])
            total_chars += msg_len

        return "\n".join(context)


class SemanticMemory(Memory):
    """Memoria semantica (long-term) con vector store."""

    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.id_counter = 0

    def add(self, content: str, metadata: dict = None) -> None:
        embedding = self.embedding_model.encode([content])[0]

        self.vector_store.add(
            ids=[str(self.id_counter)],
            embeddings=np.array([embedding]),
            metadatas=[metadata or {}],
            contents=[content]
        )
        self.id_counter += 1

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.vector_store.search(query_embedding, k=k)

        return [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score
            }
            for r in results
        ]

    def clear(self) -> None:
        # Implementation depends on vector store
        pass


class EntityMemory(Memory):
    """Memoria de entidades extraidas de conversaciones."""

    def __init__(self, llm):
        self.llm = llm
        self.entities: dict[str, dict] = {}

    def add(self, content: str, metadata: dict = None) -> None:
        # Extract entities from content
        prompt = f"""Extract named entities from this text.
Format: entity_name: entity_type: description

Text: {content}

Entities:"""

        response = self.llm.generate(prompt)

        for line in response.strip().split("\n"):
            if ":" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    entity_type = parts[1].strip() if len(parts) > 1 else "unknown"
                    description = parts[2].strip() if len(parts) > 2 else ""

                    if name not in self.entities:
                        self.entities[name] = {
                            "type": entity_type,
                            "description": description,
                            "mentions": []
                        }
                    self.entities[name]["mentions"].append(content[:100])

    def search(self, query: str, k: int = 5) -> list[dict]:
        # Find relevant entities
        results = []
        query_lower = query.lower()

        for name, info in self.entities.items():
            if name.lower() in query_lower:
                results.append({
                    "entity": name,
                    **info
                })

        return results[:k]

    def clear(self) -> None:
        self.entities = {}

    def get_entity(self, name: str) -> Optional[dict]:
        return self.entities.get(name)


class CompositeMemory:
    """Combina multiples tipos de memoria."""

    def __init__(
        self,
        conversation_memory: ConversationMemory,
        semantic_memory: SemanticMemory,
        entity_memory: Optional[EntityMemory] = None
    ):
        self.conversation = conversation_memory
        self.semantic = semantic_memory
        self.entity = entity_memory

    def add(self, content: str, metadata: dict = None) -> None:
        self.conversation.add(content, metadata)
        self.semantic.add(content, metadata)
        if self.entity:
            self.entity.add(content, metadata)

    def get_relevant_context(self, query: str) -> dict:
        """Obtiene contexto relevante de todas las memorias."""
        return {
            "recent_conversation": self.conversation.get_context(),
            "relevant_memories": self.semantic.search(query, k=5),
            "entities": self.entity.search(query) if self.entity else []
        }
```

---

## 7. Frameworks: LangChain y LlamaIndex {#frameworks}

### LangChain Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory


def create_security_agent():
    """Crea un agente de seguridad con LangChain."""

    # LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Tools
    def search_cve(cve_id: str) -> str:
        """Search for CVE information."""
        return f"CVE {cve_id}: SQL Injection vulnerability, CVSS 8.5"

    def scan_host(target: str, ports: str = "1-1000") -> str:
        """Scan a host for open ports."""
        return f"Scan of {target}: Ports 22, 80, 443 open"

    def check_reputation(ip: str) -> str:
        """Check IP reputation."""
        return f"IP {ip}: Clean reputation, no malicious activity"

    tools = [
        Tool(
            name="search_cve",
            func=search_cve,
            description="Search for CVE vulnerability information. Input: CVE ID"
        ),
        Tool(
            name="scan_host",
            func=scan_host,
            description="Scan a host for open ports. Input: hostname or IP"
        ),
        Tool(
            name="check_reputation",
            func=check_reputation,
            description="Check IP reputation. Input: IP address"
        )
    ]

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a security analyst assistant. Help users investigate
security issues, scan systems, and research vulnerabilities.
Always explain your findings and provide actionable recommendations."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

    return agent_executor


# Uso
# agent = create_security_agent()
# response = agent.invoke({"input": "Scan 192.168.1.1 and check its reputation"})
```

### LlamaIndex Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool


def create_llamaindex_agent():
    """Crea un agente con LlamaIndex."""

    # LLM
    llm = OpenAI(model="gpt-4", temperature=0)

    # Tools
    def analyze_log(log_content: str) -> str:
        """Analyze security log for threats."""
        return f"Analysis: Found 3 suspicious entries in log"

    def generate_report(findings: str) -> str:
        """Generate security report."""
        return f"Report generated with findings: {findings}"

    tools = [
        FunctionTool.from_defaults(
            fn=analyze_log,
            name="analyze_log",
            description="Analyze security logs for threats"
        ),
        FunctionTool.from_defaults(
            fn=generate_report,
            name="generate_report",
            description="Generate a security report"
        )
    ]

    # Create agent
    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        max_iterations=10
    )

    return agent
```

---

## 8. Multi-Agent Systems {#multi-agent}

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class AgentRole(Enum):
    COORDINATOR = "coordinator"
    ANALYST = "analyst"
    SCANNER = "scanner"
    REPORTER = "reporter"


@dataclass
class AgentMessage:
    """Mensaje entre agentes."""
    from_agent: str
    to_agent: str
    content: str
    message_type: str = "task"  # task, result, query


class MultiAgentSystem:
    """Sistema multi-agente para tareas de seguridad."""

    def __init__(self, llm):
        self.llm = llm
        self.agents: dict[str, Agent] = {}
        self.message_queue: list[AgentMessage] = []

    def add_agent(self, name: str, role: AgentRole, tools: list[Tool]) -> None:
        """Anade un agente especializado."""
        system_prompts = {
            AgentRole.COORDINATOR: "You coordinate security investigations and delegate tasks.",
            AgentRole.ANALYST: "You analyze threats and vulnerabilities in detail.",
            AgentRole.SCANNER: "You perform active scanning and reconnaissance.",
            AgentRole.REPORTER: "You compile findings into reports."
        }

        agent = Agent(
            llm=self.llm,
            tools=tools,
            system_prompt=system_prompts[role]
        )
        self.agents[name] = agent

    def send_message(self, message: AgentMessage) -> None:
        """Envia mensaje a un agente."""
        self.message_queue.append(message)

    def process_messages(self) -> list[str]:
        """Procesa cola de mensajes."""
        results = []

        while self.message_queue:
            msg = self.message_queue.pop(0)

            if msg.to_agent in self.agents:
                agent = self.agents[msg.to_agent]
                result = agent.run(msg.content)
                results.append(f"{msg.to_agent}: {result}")

                # Check if result should go to another agent
                if msg.message_type == "task":
                    # Report result back to sender
                    self.send_message(AgentMessage(
                        from_agent=msg.to_agent,
                        to_agent=msg.from_agent,
                        content=result,
                        message_type="result"
                    ))

        return results

    def run_investigation(self, task: str) -> str:
        """Ejecuta una investigacion con multiples agentes."""
        # Coordinator planifica
        coordinator = self.agents.get("coordinator")
        if not coordinator:
            return "No coordinator agent"

        plan = coordinator.run(f"Plan an investigation for: {task}")

        # Delegar a otros agentes
        for agent_name in ["analyst", "scanner"]:
            if agent_name in self.agents:
                self.send_message(AgentMessage(
                    from_agent="coordinator",
                    to_agent=agent_name,
                    content=f"Your part of the investigation: {plan}",
                    message_type="task"
                ))

        # Procesar y recopilar resultados
        results = self.process_messages()

        # Reporter compila
        reporter = self.agents.get("reporter")
        if reporter:
            final_report = reporter.run(
                f"Compile this investigation report:\nPlan: {plan}\nResults: {results}"
            )
            return final_report

        return "\n".join(results)
```

---

## 9. Aplicaciones en Ciberseguridad {#ciberseguridad}

```python
class SecurityAgent:
    """Agente especializado en ciberseguridad."""

    def __init__(self, llm):
        self.llm = llm

        # Security-specific tools
        self.tools = [
            Tool(
                name="threat_intel_lookup",
                description="Look up threat intelligence for an IP, domain, or hash",
                parameters={"indicator": {"type": "string"}},
                function=self._threat_intel_lookup
            ),
            Tool(
                name="vulnerability_scan",
                description="Scan a target for vulnerabilities",
                parameters={"target": {"type": "string"}},
                function=self._vulnerability_scan
            ),
            Tool(
                name="log_analysis",
                description="Analyze security logs for anomalies",
                parameters={"logs": {"type": "string"}},
                function=self._log_analysis
            ),
            Tool(
                name="generate_detection_rule",
                description="Generate detection rule (Sigma/YARA)",
                parameters={"threat_description": {"type": "string"}, "format": {"type": "string"}},
                function=self._generate_detection_rule
            ),
            Tool(
                name="incident_guidance",
                description="Get incident response guidance",
                parameters={"incident_type": {"type": "string"}},
                function=self._incident_guidance
            )
        ]

        self.agent = ReActAgent(llm, self.tools)

    def _threat_intel_lookup(self, indicator: str) -> dict:
        """Simulated threat intel lookup."""
        # In production, query actual TI feeds
        return {
            "indicator": indicator,
            "malicious": True,
            "tags": ["malware", "c2"],
            "first_seen": "2024-01-01",
            "reports": ["APT29 campaign"]
        }

    def _vulnerability_scan(self, target: str) -> dict:
        """Simulated vulnerability scan."""
        return {
            "target": target,
            "vulnerabilities": [
                {"cve": "CVE-2024-1234", "severity": "HIGH"},
                {"cve": "CVE-2024-5678", "severity": "MEDIUM"}
            ],
            "open_ports": [22, 80, 443]
        }

    def _log_analysis(self, logs: str) -> dict:
        """Analyze logs with LLM."""
        prompt = f"""Analyze these security logs for anomalies and threats:

{logs}

Identify:
1. Suspicious patterns
2. Potential attacks
3. IOCs
4. Recommended actions"""

        analysis = self.llm.generate(prompt)
        return {"analysis": analysis}

    def _generate_detection_rule(self, threat_description: str, format: str = "sigma") -> str:
        """Generate detection rule."""
        prompt = f"""Generate a {format.upper()} detection rule for:
{threat_description}

Provide a complete, valid {format} rule."""

        return self.llm.generate(prompt)

    def _incident_guidance(self, incident_type: str) -> str:
        """Get incident response guidance."""
        playbooks = {
            "ransomware": """
1. Isolate affected systems
2. Preserve evidence
3. Identify patient zero
4. Check backup integrity
5. Begin recovery
6. Report to authorities""",
            "phishing": """
1. Identify affected users
2. Reset compromised credentials
3. Block malicious indicators
4. Scan for follow-on activity
5. User awareness communication""",
            "data_breach": """
1. Contain the breach
2. Assess scope
3. Notify legal/compliance
4. Preserve evidence
5. Notify affected parties
6. Remediate vulnerabilities"""
        }
        return playbooks.get(incident_type.lower(), "No playbook found")

    def investigate(self, query: str) -> str:
        """Run security investigation."""
        return self.agent.run(query)


# Ejemplo de uso
def demo_security_agent():
    """Demo del agente de seguridad."""

    # Mock LLM para demo
    class MockLLM:
        def generate(self, prompt, **kwargs):
            if "threat intel" in prompt.lower():
                return "TOOL: threat_intel_lookup\nARGS: {\"indicator\": \"185.220.101.1\"}"
            return "ANSWER: Investigation complete. Found malicious IP associated with APT group."

    llm = MockLLM()
    agent = SecurityAgent(llm)

    # Run investigation
    result = agent.investigate("Investigate IP 185.220.101.1 for potential C2 activity")
    print(result)


if __name__ == "__main__":
    demo_security_agent()
```

---

## Resumen

Este capitulo cubrio:

1. **Arquitectura de Agentes**: Loop perceive-think-act
2. **Function Calling**: Mecanismo estandar de OpenAI
3. **ReAct**: Razonamiento alternado con acciones
4. **Planning**: Descomposicion de tareas complejas
5. **Memory**: Short-term y long-term
6. **Frameworks**: LangChain y LlamaIndex
7. **Multi-Agent**: Sistemas de agentes colaborativos
8. **Ciberseguridad**: Agentes especializados

### Recursos

- LangChain: https://langchain.com
- LlamaIndex: https://llamaindex.ai
- AutoGen: https://microsoft.github.io/autogen/
- CrewAI: https://www.crewai.com

---

*Siguiente: [75. Inference Optimization](./75-inference-optimization.md)*
