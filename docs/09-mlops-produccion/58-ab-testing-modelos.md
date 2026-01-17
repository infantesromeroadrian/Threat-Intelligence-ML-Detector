# A/B Testing y Estrategias de Deployment de Modelos

## 1. Introduccion al A/B Testing para ML

### Por que A/B Testing para Modelos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                POR QUE A/B TESTING PARA MODELOS ML                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEMA: Las metricas offline NO garantizan performance online            │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  Modelo A (Offline):                    Modelo A (Production):              │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │ F1: 0.92           │      →         │ F1: 0.85           │             │
│  │ AUC: 0.95          │      →         │ False Positives: ↑  │             │
│  │ Precision: 0.90    │                │ User complaints: ↑  │             │
│  └─────────────────────┘               └─────────────────────┘             │
│                                                                             │
│  CAUSAS DE DISCREPANCIA:                                                    │
│  • Data drift entre test set y produccion                                  │
│  • Diferencias en distribucion de usuarios                                  │
│  • Efectos de feedback loops                                                │
│  • Latencia afectando experiencia                                          │
│  • Edge cases no cubiertos en test                                         │
│                                                                             │
│  SOLUCION: Validar en produccion ANTES de rollout completo                 │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │     Traffic ──────────────────────────────────────────────────►    │   │
│  │         │                                                           │   │
│  │         │  ┌──────────────────┐     ┌──────────────────┐           │   │
│  │    90%  │  │                  │     │                  │           │   │
│  │    ─────┼─►│   Model A        │     │   Compare        │           │   │
│  │         │  │   (Control)      │────►│   Metrics        │           │   │
│  │    10%  │  │                  │     │                  │           │   │
│  │    ─────┼─►│   Model B        │     │   Statistical    │           │   │
│  │         │  │   (Variant)      │────►│   Significance   │           │   │
│  │         │  └──────────────────┘     └──────────────────┘           │   │
│  │                                              │                      │   │
│  │                                              ▼                      │   │
│  │                                     ┌──────────────────┐           │   │
│  │                                     │  Promote or      │           │   │
│  │                                     │  Rollback        │           │   │
│  │                                     └──────────────────┘           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Estrategias de Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESTRATEGIAS DE DEPLOYMENT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SHADOW MODE (Dark Launch)                                               │
│  ═══════════════════════════════════════════════════════════════════════   │
│  Nuevo modelo recibe trafico pero NO afecta usuarios                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │     Request ──────► Model A (Production) ──────► Response          │   │
│  │         │                                                           │   │
│  │         └─────────► Model B (Shadow) ──────► Log only (no response)│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Ventajas: Zero risk, compara predicciones                                  │
│  Desventajas: No mide impacto real en usuarios, doble compute              │
│                                                                             │
│  2. CANARY DEPLOYMENT                                                       │
│  ═══════════════════════════════════════════════════════════════════════   │
│  Nuevo modelo recibe pequeno % de trafico real                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │           ┌──────────────────────────────────────┐                 │   │
│  │    95%    │           Model A (Production)       │                 │   │
│  │   ────────►                                      │                 │   │
│  │           └──────────────────────────────────────┘                 │   │
│  │                                                                     │   │
│  │    5%     ┌──────────────────────────────────────┐                 │   │
│  │   ────────►          Model B (Canary)           │◄── Monitor      │   │
│  │           └──────────────────────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Ventajas: Mide impacto real, limita blast radius                          │
│  Desventajas: Requiere buen monitoring, puede afectar algunos usuarios     │
│                                                                             │
│  3. A/B TEST                                                                │
│  ═══════════════════════════════════════════════════════════════════════   │
│  Split aleatorio de trafico con medicion estadistica                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │    50%    ┌──────────────────────────────────────┐                 │   │
│  │   ────────►          Model A (Control)          ├──┐              │   │
│  │           └──────────────────────────────────────┘  │              │   │
│  │                                                     ├─► Stats Test │   │
│  │    50%    ┌──────────────────────────────────────┐  │              │   │
│  │   ────────►          Model B (Treatment)        ├──┘              │   │
│  │           └──────────────────────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Ventajas: Rigor estadistico, causa-efecto claro                           │
│  Desventajas: Requiere mas trafico, tiempo para significancia              │
│                                                                             │
│  4. MULTI-ARMED BANDIT                                                      │
│  ═══════════════════════════════════════════════════════════════════════   │
│  Asignacion dinamica basada en performance                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │           ┌──────────────────────────────────────┐                 │   │
│  │    ?%     │           Model A                    │◄──┐            │   │
│  │   ────────►                                      │   │            │   │
│  │           └──────────────────────────────────────┘   │            │   │
│  │                                                      │ Feedback   │   │
│  │    ?%     ┌──────────────────────────────────────┐   │            │   │
│  │   ────────►          Model B                     │◄──┤            │   │
│  │           └──────────────────────────────────────┘   │            │   │
│  │                                                      │            │   │
│  │    ?%     ┌──────────────────────────────────────┐   │            │   │
│  │   ────────►          Model C                     │◄──┘            │   │
│  │           └──────────────────────────────────────┘                 │   │
│  │                                                                     │   │
│  │  El algoritmo aprende y envia mas trafico al mejor modelo          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Ventajas: Minimiza regret, adaptativo                                     │
│  Desventajas: Mas complejo, menos rigor estadistico                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Implementacion de Shadow Mode

### Arquitectura Shadow Mode

```python
"""
Implementacion de Shadow Mode para validacion de modelos.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import asyncio
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ShadowPrediction:
    """Prediccion en modo shadow."""
    request_id: str
    timestamp: datetime
    features: Dict[str, Any]
    production_prediction: Any
    shadow_prediction: Any
    production_latency_ms: float
    shadow_latency_ms: float
    agreement: bool


class ShadowModeService:
    """
    Servicio de Shadow Mode para validar nuevos modelos.

    El modelo shadow recibe el mismo trafico que produccion
    pero sus predicciones no se usan para responder al usuario.
    """

    def __init__(
        self,
        production_model: Callable,
        shadow_model: Callable,
        log_predictions: bool = True,
        max_shadow_latency_ms: float = 100.0
    ):
        """
        Args:
            production_model: Modelo en produccion (callable)
            shadow_model: Modelo shadow (callable)
            log_predictions: Si guardar predicciones para analisis
            max_shadow_latency_ms: Latencia maxima para shadow (no bloquear)
        """
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.log_predictions = log_predictions
        self.max_shadow_latency_ms = max_shadow_latency_ms

        self.predictions_log: List[ShadowPrediction] = []

        # Metricas
        self.total_requests = 0
        self.agreements = 0
        self.shadow_failures = 0
        self.shadow_timeouts = 0

    async def predict(
        self,
        features: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> Any:
        """
        Hace prediccion en produccion y shadow.

        Args:
            features: Features de entrada
            request_id: ID de request para tracking

        Returns:
            Prediccion del modelo de produccion
        """
        import time
        import uuid

        request_id = request_id or str(uuid.uuid4())
        self.total_requests += 1

        # Prediccion de produccion (sincrona, critica)
        prod_start = time.time()
        production_prediction = self.production_model(features)
        prod_latency = (time.time() - prod_start) * 1000

        # Prediccion shadow (asincrona, no bloquea)
        shadow_prediction = None
        shadow_latency = 0

        try:
            shadow_start = time.time()

            # Timeout para no afectar latencia
            shadow_prediction = await asyncio.wait_for(
                asyncio.to_thread(self.shadow_model, features),
                timeout=self.max_shadow_latency_ms / 1000
            )

            shadow_latency = (time.time() - shadow_start) * 1000

        except asyncio.TimeoutError:
            self.shadow_timeouts += 1
            logger.warning(f"Shadow prediction timed out for {request_id}")

        except Exception as e:
            self.shadow_failures += 1
            logger.error(f"Shadow prediction failed: {e}")

        # Comparar predicciones
        agreement = self._compare_predictions(production_prediction, shadow_prediction)
        if agreement:
            self.agreements += 1

        # Log
        if self.log_predictions:
            self._log_prediction(ShadowPrediction(
                request_id=request_id,
                timestamp=datetime.now(),
                features=features,
                production_prediction=production_prediction,
                shadow_prediction=shadow_prediction,
                production_latency_ms=prod_latency,
                shadow_latency_ms=shadow_latency,
                agreement=agreement
            ))

        # Siempre retornar prediccion de produccion
        return production_prediction

    def _compare_predictions(
        self,
        prod_pred: Any,
        shadow_pred: Any
    ) -> bool:
        """Compara predicciones."""
        if shadow_pred is None:
            return False

        # Para clasificacion
        if isinstance(prod_pred, (int, bool)):
            return prod_pred == shadow_pred

        # Para regresion (tolerancia de 5%)
        if isinstance(prod_pred, float):
            return abs(prod_pred - shadow_pred) / (abs(prod_pred) + 1e-10) < 0.05

        # Para probabilidades
        if isinstance(prod_pred, dict) and isinstance(shadow_pred, dict):
            # Misma clase predicha
            prod_class = max(prod_pred, key=prod_pred.get)
            shadow_class = max(shadow_pred, key=shadow_pred.get)
            return prod_class == shadow_class

        return prod_pred == shadow_pred

    def _log_prediction(self, prediction: ShadowPrediction) -> None:
        """Guarda prediccion en log."""
        self.predictions_log.append(prediction)

        # En produccion: enviar a data warehouse
        # await self.send_to_warehouse(prediction)

    def get_comparison_report(self) -> Dict[str, Any]:
        """Genera reporte de comparacion."""
        if self.total_requests == 0:
            return {"error": "No requests yet"}

        agreement_rate = self.agreements / self.total_requests

        # Analizar desacuerdos
        disagreements = [p for p in self.predictions_log if not p.agreement]

        # Latencias
        prod_latencies = [p.production_latency_ms for p in self.predictions_log]
        shadow_latencies = [p.shadow_latency_ms for p in self.predictions_log if p.shadow_latency_ms > 0]

        import numpy as np

        return {
            "total_requests": self.total_requests,
            "agreement_rate": agreement_rate,
            "disagreements": len(disagreements),
            "shadow_failures": self.shadow_failures,
            "shadow_timeouts": self.shadow_timeouts,
            "production_latency": {
                "mean_ms": np.mean(prod_latencies),
                "p95_ms": np.percentile(prod_latencies, 95),
                "p99_ms": np.percentile(prod_latencies, 99)
            },
            "shadow_latency": {
                "mean_ms": np.mean(shadow_latencies) if shadow_latencies else 0,
                "p95_ms": np.percentile(shadow_latencies, 95) if shadow_latencies else 0,
                "p99_ms": np.percentile(shadow_latencies, 99) if shadow_latencies else 0
            },
            "recommendation": self._get_recommendation(agreement_rate)
        }

    def _get_recommendation(self, agreement_rate: float) -> str:
        """Genera recomendacion basada en resultados."""
        if agreement_rate >= 0.99:
            return "READY: Shadow model matches production. Safe to promote."
        elif agreement_rate >= 0.95:
            return "REVIEW: Minor differences. Review disagreements before promoting."
        elif agreement_rate >= 0.90:
            return "CAUTION: Significant differences. Investigate before promoting."
        else:
            return "NOT READY: Major differences. Do not promote to production."


# Ejemplo de uso
async def shadow_mode_example():
    """Ejemplo de shadow mode para detector de amenazas."""

    # Modelos simulados
    def production_model(features: Dict) -> int:
        """Modelo actual en produccion."""
        # Simular prediccion
        return 1 if features.get("risk_score", 0) > 0.5 else 0

    def shadow_model(features: Dict) -> int:
        """Modelo nuevo a validar."""
        # Modelo ligeramente diferente
        return 1 if features.get("risk_score", 0) > 0.45 else 0

    # Crear servicio
    service = ShadowModeService(
        production_model=production_model,
        shadow_model=shadow_model,
        max_shadow_latency_ms=50
    )

    # Simular trafico
    import random

    for i in range(1000):
        features = {
            "risk_score": random.random(),
            "bytes_sent": random.randint(100, 10000),
            "session_duration": random.randint(10, 3600)
        }

        await service.predict(features)

    # Reporte
    report = service.get_comparison_report()
    print(json.dumps(report, indent=2))

    return report
```

## 3. Canary Deployment

### Implementacion de Canary

```python
"""
Implementacion de Canary Deployment para modelos ML.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import random
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanaryStatus(Enum):
    """Estados del canary."""
    PENDING = "pending"
    RUNNING = "running"
    PROMOTING = "promoting"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CanaryMetrics:
    """Metricas del canary."""
    requests: int = 0
    errors: int = 0
    latency_sum_ms: float = 0
    predictions: Dict[str, int] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        return self.errors / max(self.requests, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_sum_ms / max(self.requests, 1)


@dataclass
class CanaryConfig:
    """Configuracion del canary."""
    initial_percentage: float = 5.0
    max_percentage: float = 50.0
    increment_percentage: float = 5.0
    increment_interval_minutes: int = 10
    min_requests_per_stage: int = 100
    max_error_rate: float = 0.05
    max_latency_ms: float = 100.0
    max_latency_increase_percent: float = 20.0


class CanaryDeployment:
    """
    Gestor de Canary Deployment para modelos ML.

    Incrementa gradualmente el trafico al nuevo modelo
    mientras monitorea metricas clave.
    """

    def __init__(
        self,
        production_model: Callable,
        canary_model: Callable,
        config: Optional[CanaryConfig] = None
    ):
        """
        Args:
            production_model: Modelo en produccion
            canary_model: Modelo canary (nuevo)
            config: Configuracion del canary
        """
        self.production_model = production_model
        self.canary_model = canary_model
        self.config = config or CanaryConfig()

        self.status = CanaryStatus.PENDING
        self.current_percentage = 0.0
        self.start_time: Optional[datetime] = None
        self.last_increment_time: Optional[datetime] = None

        self.production_metrics = CanaryMetrics()
        self.canary_metrics = CanaryMetrics()

        # Historial para analisis
        self.stage_history: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Inicia el canary deployment."""
        self.status = CanaryStatus.RUNNING
        self.current_percentage = self.config.initial_percentage
        self.start_time = datetime.now()
        self.last_increment_time = datetime.now()

        logger.info(f"Canary started at {self.current_percentage}%")

    def predict(self, features: Dict[str, Any]) -> Any:
        """
        Hace prediccion usando canary routing.

        Args:
            features: Features de entrada

        Returns:
            Prediccion (del modelo seleccionado)
        """
        if self.status != CanaryStatus.RUNNING:
            # Si no esta corriendo, usar produccion
            return self._predict_production(features)

        # Decidir modelo
        use_canary = random.random() * 100 < self.current_percentage

        if use_canary:
            return self._predict_canary(features)
        else:
            return self._predict_production(features)

    def _predict_production(self, features: Dict[str, Any]) -> Any:
        """Prediccion con modelo de produccion."""
        start = time.time()
        try:
            prediction = self.production_model(features)
            latency = (time.time() - start) * 1000

            self.production_metrics.requests += 1
            self.production_metrics.latency_sum_ms += latency

            return prediction

        except Exception as e:
            self.production_metrics.errors += 1
            raise

    def _predict_canary(self, features: Dict[str, Any]) -> Any:
        """Prediccion con modelo canary."""
        start = time.time()
        try:
            prediction = self.canary_model(features)
            latency = (time.time() - start) * 1000

            self.canary_metrics.requests += 1
            self.canary_metrics.latency_sum_ms += latency

            return prediction

        except Exception as e:
            self.canary_metrics.errors += 1
            # En canary, podemos decidir si fallar o fallback a produccion
            logger.warning(f"Canary prediction failed: {e}")
            raise

    def check_and_update(self) -> Dict[str, Any]:
        """
        Verifica metricas y actualiza estado del canary.

        Llamar periodicamente (ej: cada minuto).

        Returns:
            Dict con estado y acciones tomadas
        """
        if self.status != CanaryStatus.RUNNING:
            return {"status": self.status.value, "action": "none"}

        result = {"status": self.status.value, "action": "none"}

        # Verificar metricas
        health_check = self._check_canary_health()
        result["health_check"] = health_check

        if not health_check["healthy"]:
            # Rollback automatico
            self._rollback(health_check["reason"])
            result["action"] = "rollback"
            result["reason"] = health_check["reason"]
            return result

        # Verificar si es tiempo de incrementar
        time_since_increment = datetime.now() - self.last_increment_time
        min_requests_met = self.canary_metrics.requests >= self.config.min_requests_per_stage

        if (time_since_increment >= timedelta(minutes=self.config.increment_interval_minutes)
            and min_requests_met):

            if self.current_percentage >= self.config.max_percentage:
                # Listo para promover
                self._promote()
                result["action"] = "promote"
            else:
                # Incrementar porcentaje
                self._increment_percentage()
                result["action"] = "increment"
                result["new_percentage"] = self.current_percentage

        return result

    def _check_canary_health(self) -> Dict[str, Any]:
        """Verifica salud del canary."""
        result = {"healthy": True, "checks": []}

        # Check 1: Error rate
        if self.canary_metrics.requests >= 10:
            canary_error_rate = self.canary_metrics.error_rate
            prod_error_rate = self.production_metrics.error_rate

            if canary_error_rate > self.config.max_error_rate:
                result["healthy"] = False
                result["reason"] = f"Canary error rate too high: {canary_error_rate:.2%}"
                return result

            result["checks"].append({
                "name": "error_rate",
                "canary": canary_error_rate,
                "production": prod_error_rate,
                "passed": True
            })

        # Check 2: Latency
        if self.canary_metrics.requests >= 10:
            canary_latency = self.canary_metrics.avg_latency_ms
            prod_latency = self.production_metrics.avg_latency_ms

            if canary_latency > self.config.max_latency_ms:
                result["healthy"] = False
                result["reason"] = f"Canary latency too high: {canary_latency:.1f}ms"
                return result

            latency_increase = (canary_latency - prod_latency) / max(prod_latency, 1) * 100
            if latency_increase > self.config.max_latency_increase_percent:
                result["healthy"] = False
                result["reason"] = f"Canary latency increase: {latency_increase:.1f}%"
                return result

            result["checks"].append({
                "name": "latency",
                "canary_ms": canary_latency,
                "production_ms": prod_latency,
                "passed": True
            })

        return result

    def _increment_percentage(self) -> None:
        """Incrementa porcentaje del canary."""
        old_percentage = self.current_percentage
        self.current_percentage = min(
            self.current_percentage + self.config.increment_percentage,
            self.config.max_percentage
        )
        self.last_increment_time = datetime.now()

        # Reset metricas para nueva etapa
        self.stage_history.append({
            "percentage": old_percentage,
            "production_metrics": {
                "requests": self.production_metrics.requests,
                "error_rate": self.production_metrics.error_rate,
                "avg_latency_ms": self.production_metrics.avg_latency_ms
            },
            "canary_metrics": {
                "requests": self.canary_metrics.requests,
                "error_rate": self.canary_metrics.error_rate,
                "avg_latency_ms": self.canary_metrics.avg_latency_ms
            }
        })

        self.canary_metrics = CanaryMetrics()

        logger.info(f"Canary percentage increased: {old_percentage}% -> {self.current_percentage}%")

    def _promote(self) -> None:
        """Promueve canary a produccion."""
        self.status = CanaryStatus.PROMOTING
        self.current_percentage = 100.0

        logger.info("Canary promoted to production!")

        # En produccion real: actualizar routing rules, model registry, etc.
        self.status = CanaryStatus.COMPLETED

    def _rollback(self, reason: str) -> None:
        """Rollback a produccion."""
        self.status = CanaryStatus.ROLLING_BACK
        self.current_percentage = 0.0

        logger.error(f"Canary rollback: {reason}")

        self.status = CanaryStatus.FAILED

    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del canary."""
        return {
            "status": self.status.value,
            "current_percentage": self.current_percentage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_minutes": (
                (datetime.now() - self.start_time).total_seconds() / 60
                if self.start_time else 0
            ),
            "production_metrics": {
                "requests": self.production_metrics.requests,
                "error_rate": self.production_metrics.error_rate,
                "avg_latency_ms": self.production_metrics.avg_latency_ms
            },
            "canary_metrics": {
                "requests": self.canary_metrics.requests,
                "error_rate": self.canary_metrics.error_rate,
                "avg_latency_ms": self.canary_metrics.avg_latency_ms
            },
            "stage_history": self.stage_history
        }
```

## 4. A/B Testing con Significancia Estadistica

### Framework de A/B Testing

```python
"""
A/B Testing estadisticamente riguroso para modelos ML.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from scipy import stats
import hashlib


class TestStatus(Enum):
    """Estados del test."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    STOPPED_EARLY = "stopped_early"


@dataclass
class ABTestConfig:
    """Configuracion del A/B test."""
    control_model_name: str
    treatment_model_name: str
    traffic_split: float = 0.5  # Porcentaje para treatment
    min_sample_size: int = 1000
    significance_level: float = 0.05  # alpha
    power: float = 0.8  # 1 - beta
    minimum_detectable_effect: float = 0.02  # MDE en metric
    primary_metric: str = "conversion"  # Metrica principal


@dataclass
class VariantMetrics:
    """Metricas de una variante."""
    name: str
    samples: int = 0
    successes: int = 0  # Para metricas binarias
    sum_value: float = 0  # Para metricas continuas
    sum_squared: float = 0  # Para calcular varianza

    @property
    def mean(self) -> float:
        return self.sum_value / max(self.samples, 1)

    @property
    def rate(self) -> float:
        """Para metricas binarias (conversion rate, etc.)"""
        return self.successes / max(self.samples, 1)

    @property
    def variance(self) -> float:
        if self.samples < 2:
            return 0
        return (self.sum_squared - (self.sum_value ** 2) / self.samples) / (self.samples - 1)

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


class ABTestFramework:
    """
    Framework de A/B Testing para modelos ML.

    Soporta:
    - Assignment deterministico (por user_id)
    - Metricas binarias y continuas
    - Tests de significancia (t-test, chi-squared)
    - Early stopping con sequential testing
    """

    def __init__(self, config: ABTestConfig):
        """
        Args:
            config: Configuracion del test
        """
        self.config = config
        self.status = TestStatus.NOT_STARTED
        self.start_time: Optional[datetime] = None

        self.control = VariantMetrics(name=config.control_model_name)
        self.treatment = VariantMetrics(name=config.treatment_model_name)

        # Para sequential testing
        self.interim_analyses: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Inicia el A/B test."""
        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()

    def assign_variant(self, user_id: str) -> str:
        """
        Asigna variante a usuario de forma deterministica.

        El mismo user_id siempre recibe la misma variante.

        Args:
            user_id: ID del usuario

        Returns:
            Nombre del modelo asignado
        """
        # Hash deterministico
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        threshold = self.config.traffic_split * (2 ** 128)

        if hash_value < threshold:
            return self.config.treatment_model_name
        else:
            return self.config.control_model_name

    def record_outcome(
        self,
        user_id: str,
        variant: str,
        success: bool = None,
        value: float = None
    ) -> None:
        """
        Registra resultado de un usuario.

        Args:
            user_id: ID del usuario
            variant: Variante asignada
            success: Para metricas binarias (conversion, click, etc.)
            value: Para metricas continuas (revenue, engagement, etc.)
        """
        if variant == self.config.control_model_name:
            metrics = self.control
        else:
            metrics = self.treatment

        metrics.samples += 1

        if success is not None:
            if success:
                metrics.successes += 1
            metrics.sum_value += float(success)
            metrics.sum_squared += float(success) ** 2

        if value is not None:
            metrics.sum_value += value
            metrics.sum_squared += value ** 2

    def calculate_sample_size(
        self,
        baseline_rate: float,
        mde: float = None
    ) -> int:
        """
        Calcula tamano de muestra requerido.

        Args:
            baseline_rate: Tasa baseline (conversion rate actual)
            mde: Minimum detectable effect

        Returns:
            Tamano de muestra por variante
        """
        mde = mde or self.config.minimum_detectable_effect
        alpha = self.config.significance_level
        power = self.config.power

        # Formula para two-sample proportion test
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_avg = (p1 + p2) / 2

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = (
            2 * p_avg * (1 - p_avg) * (z_alpha + z_beta) ** 2
        ) / (mde ** 2)

        return int(np.ceil(n))

    def test_significance_proportion(self) -> Dict[str, Any]:
        """
        Test de significancia para metricas binarias (proporciones).

        Usa chi-squared test o z-test para proporciones.
        """
        n1 = self.control.samples
        n2 = self.treatment.samples
        x1 = self.control.successes
        x2 = self.treatment.successes

        if n1 < 10 or n2 < 10:
            return {"error": "Not enough samples", "significant": False}

        p1 = x1 / n1
        p2 = x2 / n2
        p_pooled = (x1 + x2) / (n1 + n2)

        # Z-test para proporciones
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z_stat = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Intervalo de confianza
        se_diff = np.sqrt(p1 * (1-p1) / n1 + p2 * (1-p2) / n2)
        ci_lower = (p2 - p1) - 1.96 * se_diff
        ci_upper = (p2 - p1) + 1.96 * se_diff

        significant = p_value < self.config.significance_level

        return {
            "test_type": "proportion_z_test",
            "control_rate": p1,
            "treatment_rate": p2,
            "absolute_effect": p2 - p1,
            "relative_effect": (p2 - p1) / p1 if p1 > 0 else 0,
            "z_statistic": z_stat,
            "p_value": p_value,
            "confidence_interval": [ci_lower, ci_upper],
            "significant": significant,
            "winner": (
                self.config.treatment_model_name if significant and p2 > p1
                else self.config.control_model_name if significant and p1 > p2
                else "no_winner"
            )
        }

    def test_significance_continuous(self) -> Dict[str, Any]:
        """
        Test de significancia para metricas continuas.

        Usa Welch's t-test (no asume varianzas iguales).
        """
        n1 = self.control.samples
        n2 = self.treatment.samples

        if n1 < 10 or n2 < 10:
            return {"error": "Not enough samples", "significant": False}

        mean1 = self.control.mean
        mean2 = self.treatment.mean
        var1 = self.control.variance
        var2 = self.treatment.variance

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind_from_stats(
            mean1, np.sqrt(var1), n1,
            mean2, np.sqrt(var2), n2,
            equal_var=False
        )

        # Intervalo de confianza
        se_diff = np.sqrt(var1/n1 + var2/n2)
        df = ((var1/n1 + var2/n2)**2 /
              ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)))
        t_crit = stats.t.ppf(1 - self.config.significance_level/2, df)

        ci_lower = (mean2 - mean1) - t_crit * se_diff
        ci_upper = (mean2 - mean1) + t_crit * se_diff

        significant = p_value < self.config.significance_level

        return {
            "test_type": "welch_t_test",
            "control_mean": mean1,
            "treatment_mean": mean2,
            "absolute_effect": mean2 - mean1,
            "relative_effect": (mean2 - mean1) / mean1 if mean1 > 0 else 0,
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence_interval": [ci_lower, ci_upper],
            "significant": significant,
            "winner": (
                self.config.treatment_model_name if significant and mean2 > mean1
                else self.config.control_model_name if significant and mean1 > mean2
                else "no_winner"
            )
        }

    def get_current_results(self) -> Dict[str, Any]:
        """Obtiene resultados actuales del test."""
        # Usar test de proporcion por defecto
        significance_test = self.test_significance_proportion()

        return {
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "control": {
                "name": self.control.name,
                "samples": self.control.samples,
                "rate": self.control.rate,
                "mean": self.control.mean
            },
            "treatment": {
                "name": self.treatment.name,
                "samples": self.treatment.samples,
                "rate": self.treatment.rate,
                "mean": self.treatment.mean
            },
            "significance_test": significance_test,
            "sample_size_required": self.calculate_sample_size(self.control.rate)
            if self.control.samples > 0 else None
        }


# =============================================================================
# APLICACION A CIBERSEGURIDAD
# =============================================================================

def security_ab_test_example():
    """Ejemplo de A/B test para detector de amenazas."""

    # Configurar test
    config = ABTestConfig(
        control_model_name="threat_detector_v1",
        treatment_model_name="threat_detector_v2",
        traffic_split=0.5,
        min_sample_size=5000,
        significance_level=0.05,
        minimum_detectable_effect=0.02,  # 2% mejora en deteccion
        primary_metric="true_positive_rate"
    )

    test = ABTestFramework(config)
    test.start()

    # Simular trafico
    np.random.seed(42)

    # Tasas reales (desconocidas en practica)
    control_tpr = 0.85  # True positive rate del modelo actual
    treatment_tpr = 0.88  # True positive rate del nuevo modelo

    for i in range(10000):
        user_id = f"session_{i}"

        # Asignar variante
        variant = test.assign_variant(user_id)

        # Simular outcome
        if variant == config.control_model_name:
            # El modelo detecto la amenaza?
            success = np.random.random() < control_tpr
        else:
            success = np.random.random() < treatment_tpr

        test.record_outcome(user_id, variant, success=success)

    # Resultados
    results = test.get_current_results()

    print("=== A/B Test Results ===")
    print(f"Control ({results['control']['name']}):")
    print(f"  Samples: {results['control']['samples']}")
    print(f"  Detection Rate: {results['control']['rate']:.2%}")

    print(f"\nTreatment ({results['treatment']['name']}):")
    print(f"  Samples: {results['treatment']['samples']}")
    print(f"  Detection Rate: {results['treatment']['rate']:.2%}")

    sig = results['significance_test']
    print(f"\nSignificance Test:")
    print(f"  p-value: {sig['p_value']:.4f}")
    print(f"  Significant: {sig['significant']}")
    print(f"  Winner: {sig['winner']}")
    print(f"  Effect: {sig['absolute_effect']:.2%} ({sig['relative_effect']:.1%} relative)")

    return results
```

## 5. Multi-Armed Bandits

### Implementacion de Bandits

```python
"""
Multi-Armed Bandits para seleccion adaptativa de modelos.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


@dataclass
class ArmStats:
    """Estadisticas de un brazo (modelo)."""
    name: str
    pulls: int = 0
    total_reward: float = 0
    successes: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.pulls, 1)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.pulls, 1)


class BanditAlgorithm(ABC):
    """Algoritmo de bandit base."""

    @abstractmethod
    def select_arm(self, arms: List[ArmStats]) -> int:
        """Selecciona cual brazo tirar."""
        pass

    @abstractmethod
    def update(self, arm_index: int, reward: float) -> None:
        """Actualiza estadisticas despues de observar reward."""
        pass


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy: Explora con probabilidad epsilon, explota con 1-epsilon.

    Simple pero efectivo.
    """

    def __init__(self, epsilon: float = 0.1, decay: float = 0.999):
        """
        Args:
            epsilon: Probabilidad de explorar
            decay: Factor de decay para epsilon
        """
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self.arms_stats: List[ArmStats] = []

    def select_arm(self, arms: List[ArmStats]) -> int:
        """Selecciona brazo."""
        self.arms_stats = arms

        if np.random.random() < self.epsilon:
            # Explorar: elegir random
            return np.random.randint(len(arms))
        else:
            # Explotar: elegir mejor
            rewards = [arm.mean_reward for arm in arms]
            return int(np.argmax(rewards))

    def update(self, arm_index: int, reward: float) -> None:
        """Actualiza y decae epsilon."""
        self.epsilon *= self.decay


class UCB1(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB1).

    Balancea exploration/exploitation usando intervalos de confianza.
    Selecciona el brazo con mayor upper confidence bound.
    """

    def __init__(self, c: float = 2.0):
        """
        Args:
            c: Parametro de exploracion (mayor = mas exploracion)
        """
        self.c = c
        self.total_pulls = 0

    def select_arm(self, arms: List[ArmStats]) -> int:
        """
        Selecciona brazo con mayor UCB.

        UCB = mean_reward + c * sqrt(log(total_pulls) / arm_pulls)
        """
        self.total_pulls = sum(arm.pulls for arm in arms)

        # Asegurar que todos los brazos se prueben al menos una vez
        for i, arm in enumerate(arms):
            if arm.pulls == 0:
                return i

        ucb_values = []
        for arm in arms:
            exploitation = arm.mean_reward
            exploration = self.c * np.sqrt(np.log(self.total_pulls) / arm.pulls)
            ucb_values.append(exploitation + exploration)

        return int(np.argmax(ucb_values))

    def update(self, arm_index: int, reward: float) -> None:
        """UCB no necesita update especial."""
        pass


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling (Bayesian).

    Para rewards binarios, usa distribucion Beta.
    Muestrea de la distribucion posterior de cada brazo.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Args:
            prior_alpha: Prior Beta alpha (exitos previos + 1)
            prior_beta: Prior Beta beta (fallos previos + 1)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.arms_stats: List[ArmStats] = []

    def select_arm(self, arms: List[ArmStats]) -> int:
        """
        Muestrea de posterior Beta y selecciona el mayor.
        """
        self.arms_stats = arms

        samples = []
        for arm in arms:
            alpha = self.prior_alpha + arm.successes
            beta = self.prior_beta + (arm.pulls - arm.successes)
            sample = np.random.beta(alpha, beta)
            samples.append(sample)

        return int(np.argmax(samples))

    def update(self, arm_index: int, reward: float) -> None:
        """Thompson Sampling actualiza automaticamente con las stats."""
        pass


class MultiArmedBanditRouter:
    """
    Router de modelos usando Multi-Armed Bandits.

    Aprende en tiempo real cual modelo es mejor
    y envia mas trafico al ganador.
    """

    def __init__(
        self,
        models: Dict[str, callable],
        algorithm: BanditAlgorithm,
        reward_function: callable
    ):
        """
        Args:
            models: Dict de nombre -> modelo callable
            algorithm: Algoritmo de bandit a usar
            reward_function: Funcion que convierte outcome a reward
        """
        self.models = models
        self.algorithm = algorithm
        self.reward_function = reward_function

        # Inicializar brazos
        self.arms = [ArmStats(name=name) for name in models.keys()]
        self.arm_names = list(models.keys())

    def select_model(self) -> tuple:
        """
        Selecciona modelo para siguiente request.

        Returns:
            (arm_index, model_name, model_callable)
        """
        arm_index = self.algorithm.select_arm(self.arms)
        model_name = self.arm_names[arm_index]
        model = self.models[model_name]

        return arm_index, model_name, model

    def record_outcome(
        self,
        arm_index: int,
        outcome: Any
    ) -> float:
        """
        Registra outcome y actualiza estadisticas.

        Args:
            arm_index: Indice del brazo usado
            outcome: Resultado observado

        Returns:
            Reward calculado
        """
        reward = self.reward_function(outcome)

        arm = self.arms[arm_index]
        arm.pulls += 1
        arm.total_reward += reward
        if reward > 0.5:  # Threshold para "exito"
            arm.successes += 1

        self.algorithm.update(arm_index, reward)

        return reward

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadisticas actuales."""
        total_pulls = sum(arm.pulls for arm in self.arms)

        stats = {
            "total_requests": total_pulls,
            "arms": []
        }

        for arm in self.arms:
            stats["arms"].append({
                "name": arm.name,
                "pulls": arm.pulls,
                "traffic_share": arm.pulls / max(total_pulls, 1),
                "mean_reward": arm.mean_reward,
                "success_rate": arm.success_rate
            })

        # Ordenar por reward
        stats["arms"].sort(key=lambda x: x["mean_reward"], reverse=True)
        stats["current_best"] = stats["arms"][0]["name"] if stats["arms"] else None

        return stats


# =============================================================================
# EJEMPLO PARA CIBERSEGURIDAD
# =============================================================================

def security_bandit_example():
    """
    Ejemplo de Multi-Armed Bandit para seleccion de
    modelo de deteccion de amenazas.
    """

    # Modelos (diferentes versiones/configuraciones)
    def model_v1(features):
        """Modelo conservador (menos falsos positivos)."""
        return 1 if features["risk_score"] > 0.7 else 0

    def model_v2(features):
        """Modelo agresivo (menos falsos negativos)."""
        return 1 if features["risk_score"] > 0.4 else 0

    def model_v3(features):
        """Modelo balanceado."""
        return 1 if features["risk_score"] > 0.55 else 0

    models = {
        "detector_conservative_v1": model_v1,
        "detector_aggressive_v2": model_v2,
        "detector_balanced_v3": model_v3
    }

    # Funcion de reward
    # Para seguridad: penalizar falsos negativos mas que falsos positivos
    def security_reward(outcome: Dict) -> float:
        prediction = outcome["prediction"]
        ground_truth = outcome["ground_truth"]

        if prediction == ground_truth:
            return 1.0  # Correcto
        elif prediction == 0 and ground_truth == 1:
            return -2.0  # Falso negativo (muy malo)
        else:
            return -0.5  # Falso positivo (malo pero menos)

    # Crear router con Thompson Sampling
    router = MultiArmedBanditRouter(
        models=models,
        algorithm=ThompsonSampling(),
        reward_function=security_reward
    )

    # Simular trafico
    np.random.seed(42)

    # Tasas de precision por modelo (desconocidas en practica)
    true_performance = {
        "detector_conservative_v1": {"tp": 0.80, "fp": 0.05},
        "detector_aggressive_v2": {"tp": 0.95, "fp": 0.25},
        "detector_balanced_v3": {"tp": 0.90, "fp": 0.10}
    }

    for i in range(5000):
        # Generar features
        features = {"risk_score": np.random.random()}

        # Seleccionar modelo
        arm_index, model_name, model = router.select_model()

        # Hacer prediccion
        prediction = model(features)

        # Simular ground truth (con performance real del modelo)
        perf = true_performance[model_name]
        if features["risk_score"] > 0.5:  # Realmente malicioso
            detected = np.random.random() < perf["tp"]
        else:  # Realmente benigno
            detected = np.random.random() < perf["fp"]

        ground_truth = 1 if features["risk_score"] > 0.5 else 0

        # Registrar outcome
        outcome = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "detected": detected
        }

        router.record_outcome(arm_index, outcome)

    # Resultados
    stats = router.get_statistics()

    print("=== Multi-Armed Bandit Results ===")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Current best: {stats['current_best']}")
    print("\nArm Statistics:")
    for arm in stats["arms"]:
        print(f"  {arm['name']}:")
        print(f"    Traffic share: {arm['traffic_share']:.1%}")
        print(f"    Mean reward: {arm['mean_reward']:.3f}")
        print(f"    Success rate: {arm['success_rate']:.1%}")

    return stats
```

## 6. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               RESUMEN: A/B Testing y Deployment de Modelos                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ESTRATEGIAS DE DEPLOYMENT                                                  │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Shadow Mode: Nuevo modelo no afecta usuarios, solo logging              │
│  • Canary: Pequeno % de trafico, rollback automatico si falla              │
│  • A/B Test: Split aleatorio, significancia estadistica                    │
│  • Multi-Armed Bandit: Asignacion adaptativa, minimiza regret              │
│                                                                             │
│  SHADOW MODE                                                                │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Zero risk: predicciones no se usan                                      │
│  • Comparar predicciones prod vs shadow                                    │
│  • Validar latencia y errores                                              │
│  • Prerequisito antes de canary/A/B                                        │
│                                                                             │
│  CANARY DEPLOYMENT                                                          │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Empezar con 5% de trafico                                               │
│  • Incrementar gradualmente si metricas OK                                 │
│  • Rollback automatico si error_rate o latencia suben                      │
│  • Promover a 100% cuando llegue a max_percentage                          │
│                                                                             │
│  A/B TESTING                                                                │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Assignment deterministico por user_id                                   │
│  • Calcular sample size antes de empezar                                   │
│  • Tests: z-test (proporciones), t-test (continuas)                        │
│  • Esperar significancia estadistica antes de decidir                      │
│                                                                             │
│  MULTI-ARMED BANDITS                                                        │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Epsilon-Greedy: Simple, explora con probabilidad epsilon                │
│  • UCB1: Intervalos de confianza, balance exploration/exploitation         │
│  • Thompson Sampling: Bayesiano, muestrea de posterior                     │
│  • Envia mas trafico automaticamente al mejor modelo                       │
│                                                                             │
│  CIBERSEGURIDAD                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Preferir Shadow Mode primero (zero risk)                                │
│  • Canary con rollback rapido (segundos, no minutos)                       │
│  • A/B test: penalizar falsos negativos mas que falsos positivos           │
│  • Bandits: reward function que refleje impacto de errores                 │
│                                                                             │
│  METRICAS CLAVE                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Error rate (canary)                                                     │
│  • Latencia (p50, p95, p99)                                                │
│  • Detection rate / True positive rate                                     │
│  • False positive rate                                                     │
│  • Business metrics (si aplica)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Fin de la seccion MLOps.** Esta seccion cubre el ciclo completo desde experiment tracking hasta deployment seguro de modelos en produccion.
