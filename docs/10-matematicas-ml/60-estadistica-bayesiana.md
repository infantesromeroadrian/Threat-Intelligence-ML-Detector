# Estadística Bayesiana para Machine Learning

## 1. Introducción: Frequentistas vs Bayesianos

### El Problema de la Incertidumbre

En ML clásico (frequentista), obtenemos un único punto estimado:

```
┌─────────────────────────────────────────────────────────────┐
│  ENFOQUE FREQUENTISTA                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  θ* = argmax P(D|θ)    ← Maximum Likelihood Estimation       │
│                                                              │
│  Resultado: UN SOLO VALOR de θ                              │
│                                                              │
│  Problema: ¿Qué tan CONFIADO estoy en θ*?                   │
│            No tengo información de incertidumbre             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ENFOQUE BAYESIANO                                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  P(θ|D) = P(D|θ) · P(θ) / P(D)                              │
│                                                              │
│  Resultado: DISTRIBUCIÓN COMPLETA de θ                      │
│                                                              │
│  Beneficio: Sé qué tan seguro estoy de cada predicción      │
│             Puedo decir "predicción ± incertidumbre"        │
└─────────────────────────────────────────────────────────────┘
```

### Visualización: Punto vs Distribución

```
     FREQUENTISTA                        BAYESIANO

P(θ)│                              P(θ)│      ╱╲
    │                                  │     ╱  ╲
    │                                  │    ╱    ╲
    │       │                          │   ╱      ╲
    │       │ ← θ*                     │  ╱        ╲
    │       │                          │ ╱          ╲
    └───────●─────── θ                 └─────────────── θ
                                           ↑
    Solo un punto                     Distribución completa
    ¿Es θ* = 2.3 o 2.4?               "θ está entre 2.0 y 2.6
    No lo sé                           con 95% de confianza"
```

### Aplicaciones en ML

```
┌─────────────────────────────────────────────────────────────┐
│  ¿CUÁNDO USAR BAYESIAN ML?                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATOS ESCASOS                                            │
│     • Pocos ejemplos de entrenamiento                        │
│     • El prior aporta información crucial                    │
│     • Ej: Ensayos clínicos, A/B testing inicial             │
│                                                              │
│  2. INCERTIDUMBRE CRÍTICA                                    │
│     • Decisiones de alto riesgo                              │
│     • Necesitas saber "qué tan seguro estás"                │
│     • Ej: Diagnóstico médico, conducción autónoma           │
│                                                              │
│  3. ACTIVE LEARNING                                          │
│     • Decidir qué dato etiquetar siguiente                   │
│     • Explorar vs explotar                                   │
│     • Ej: Optimización de hiperparámetros bayesiana         │
│                                                              │
│  4. MODELOS GENERATIVOS                                      │
│     • VAEs, modelos probabilísticos                          │
│     • Sampling de distribuciones aprendidas                  │
└─────────────────────────────────────────────────────────────┘
```

## 2. Teorema de Bayes: El Fundamento

### La Fórmula Central

```
┌─────────────────────────────────────────────────────────────┐
│  TEOREMA DE BAYES                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                    P(D|θ) · P(θ)                            │
│       P(θ|D) = ─────────────────                            │
│                      P(D)                                    │
│                                                              │
│  Donde:                                                      │
│    P(θ|D)   = POSTERIOR  ← Lo que queremos                  │
│    P(D|θ)   = LIKELIHOOD ← Probabilidad de datos dado θ    │
│    P(θ)     = PRIOR      ← Creencia inicial sobre θ        │
│    P(D)     = EVIDENCE   ← Constante de normalización       │
│                                                              │
│  En palabras:                                                │
│    POSTERIOR ∝ LIKELIHOOD × PRIOR                           │
│    "Lo que creo después de ver datos" =                     │
│    "Cómo de bien explican los datos" × "Lo que creía antes"│
└─────────────────────────────────────────────────────────────┘
```

### Diagrama de Flujo Bayesiano

```
                 ┌───────────────┐
                 │    PRIOR      │
                 │    P(θ)       │
                 │ (Conocimiento │
                 │   previo)     │
                 └───────┬───────┘
                         │
                         ▼
    ┌────────────────────┴────────────────────┐
    │           REGLA DE BAYES                │
    │                                          │
    │  posterior ∝ likelihood × prior         │
    │                                          │
    └────────────────────┬────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌───────┐       ┌─────────┐      ┌───────┐
    │ DATOS │       │POSTERIOR│      │NUEVO  │
    │   D   │ ───→  │  P(θ|D) │ ───→ │ PRIOR │
    └───────┘       │(Creencia│      │       │
                    │ actual) │      │(Datos │
                    └─────────┘      │futuros)
                                     └───────┘

El posterior de hoy es el prior de mañana → Aprendizaje continuo
```

### Ejemplo Numérico: Clasificación de Spam

```
Problema: ¿Es un email spam dado que contiene "gratis"?

Datos conocidos:
  P(spam) = 0.3                    ← Prior: 30% emails son spam
  P("gratis"|spam) = 0.6           ← Likelihood si es spam
  P("gratis"|no spam) = 0.1        ← Likelihood si no es spam

Calcular P(spam|"gratis"):

PASO 1: P(D) = P("gratis")
  = P("gratis"|spam)·P(spam) + P("gratis"|no spam)·P(no spam)
  = 0.6 × 0.3 + 0.1 × 0.7
  = 0.18 + 0.07
  = 0.25

PASO 2: Bayes
  P(spam|"gratis") = P("gratis"|spam) × P(spam) / P("gratis")
                   = 0.6 × 0.3 / 0.25
                   = 0.18 / 0.25
                   = 0.72

Resultado: Si contiene "gratis", hay 72% probabilidad de spam
           (El prior era 30%, los datos lo actualizaron a 72%)
```

```python
import numpy as np

def bayes_spam_classifier():
    """Ejemplo de clasificador Naive Bayes manual."""
    # Priors
    p_spam = 0.3
    p_no_spam = 1 - p_spam

    # Likelihoods para "gratis"
    p_gratis_dado_spam = 0.6
    p_gratis_dado_no_spam = 0.1

    # Evidence (marginalizando)
    p_gratis = (p_gratis_dado_spam * p_spam +
                p_gratis_dado_no_spam * p_no_spam)

    # Posterior usando Bayes
    p_spam_dado_gratis = (p_gratis_dado_spam * p_spam) / p_gratis

    print(f"Prior P(spam): {p_spam:.2f}")
    print(f"Evidence P('gratis'): {p_gratis:.2f}")
    print(f"Posterior P(spam|'gratis'): {p_spam_dado_gratis:.2f}")

    return p_spam_dado_gratis

bayes_spam_classifier()
```

## 3. Prior, Likelihood, Posterior

### Elección del Prior

```
┌─────────────────────────────────────────────────────────────┐
│  TIPOS DE PRIORS                                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PRIOR INFORMATIVO                                           │
│    • Incorpora conocimiento experto                          │
│    • Ej: "El peso promedio de adultos es ~70kg ± 15kg"     │
│    • P(θ) = Normal(70, 15²)                                 │
│                                                              │
│  PRIOR NO INFORMATIVO (Vago)                                 │
│    • Mínima información                                      │
│    • Ej: "θ puede ser cualquier valor positivo"            │
│    • P(θ) = Uniforme(0, ∞) o Jeffreys prior                │
│                                                              │
│  PRIOR DÉBILMENTE INFORMATIVO                                │
│    • Regularización suave                                    │
│    • Ej: "Los coeficientes probablemente están cerca de 0" │
│    • P(θ) = Normal(0, 10²) ← Equivalente a Ridge           │
│                                                              │
│  PRIOR SPIKE-AND-SLAB                                        │
│    • Para sparsity (algunos θ = 0)                          │
│    • Mezcla de delta en 0 + distribución continua           │
└─────────────────────────────────────────────────────────────┘
```

### Visualización de Priors

```
    PRIOR INFORMATIVO              PRIOR NO INFORMATIVO
    (Conocimiento fuerte)          (Sin conocimiento)

P(θ)│    ╱╲                    P(θ)│─────────────────
    │   ╱  ╲                       │
    │  ╱    ╲                      │
    │ ╱      ╲                     │
    │╱        ╲                    │
    └────●────────── θ             └──────────────── θ
         ↑                         Todos los valores
    θ cercano a 70                 igualmente probables


    PRIOR DÉBIL (Ridge)            PRIOR LAPLACE (Lasso)
    Normal(0, σ²)                  Laplace(0, b)

P(θ)│    ╱╲                    P(θ)│    ╱\
    │   ╱  ╲                       │   ╱  \
    │  ╱    ╲                      │  ╱    \
    │ ╱      ╲                     │ ╱      \
    │╱        ╲                    │╱        \
    └────●────────── θ             └────●────────── θ
         ↑                              ↑
    Prefiere θ cerca de 0         Prefiere θ = 0 exacto
    (regularización L2)            (regularización L1)
```

### El Likelihood: La Voz de los Datos

```
┌─────────────────────────────────────────────────────────────┐
│  LIKELIHOOD: P(D|θ)                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  "¿Qué tan probable es ver estos datos si θ fuera cierto?" │
│                                                              │
│  Para regresión lineal con ruido Gaussiano:                 │
│                                                              │
│    y = Xθ + ε,  donde ε ~ N(0, σ²)                         │
│                                                              │
│    P(y|X, θ, σ²) = N(Xθ, σ²I)                              │
│                                                              │
│    = (2πσ²)^(-n/2) exp(-||y - Xθ||² / 2σ²)                │
│                                                              │
│  Log-likelihood:                                             │
│    log P(D|θ) = -n/2 log(2πσ²) - ||y - Xθ||² / 2σ²        │
│                                                              │
│  Maximizar log-likelihood ≡ Minimizar MSE (si σ fijo)      │
└─────────────────────────────────────────────────────────────┘
```

### Posterior: La Combinación

```
┌─────────────────────────────────────────────────────────────┐
│  EVOLUCIÓN DEL POSTERIOR CON MÁS DATOS                      │
├─────────────────────────────────────────────────────────────┤

n = 0 (sin datos)          n = 10                n = 100
Solo prior                 Prior + datos         Datos dominan

P(θ)│    ╱╲             P(θ)│   ╱╲            P(θ)│  ╱╲
    │   ╱  ╲                │  ╱  ╲               │ ╱  ╲
    │  ╱    ╲               │ ╱    ╲              │╱    ╲
    │ ╱      ╲              │╱      ╲             │      ╲
    └─────────── θ          └─────────── θ        └─────────── θ
         ↑                       ↑                    ↑
    Ancho = σ_prior         Se estrecha          Muy concentrado

A medida que n → ∞:
  • El posterior se concentra en el MLE (máxima verosimilitud)
  • La influencia del prior desaparece
  • Bayesiano ≈ Frequentista para datos abundantes

El prior importa MÁS cuando hay POCOS datos
└─────────────────────────────────────────────────────────────┘
```

## 4. Conjugate Priors: Soluciones Analíticas

### ¿Qué es un Prior Conjugado?

```
┌─────────────────────────────────────────────────────────────┐
│  CONJUGATE PRIOR                                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Un prior es CONJUGADO al likelihood si:                    │
│                                                              │
│    prior × likelihood = posterior de la MISMA FAMILIA       │
│                                                              │
│  Beneficio: El posterior tiene FORMA CERRADA               │
│             No necesitamos métodos numéricos (MCMC)         │
│                                                              │
│  Ejemplos clásicos:                                          │
│    Likelihood          Prior Conjugado      Posterior        │
│    ─────────────────────────────────────────────────────    │
│    Bernoulli(θ)        Beta(α, β)          Beta(α', β')    │
│    Poisson(λ)          Gamma(α, β)         Gamma(α', β')   │
│    Normal(μ, σ²)       Normal(μ₀, σ₀²)    Normal(μ', σ'²) │
│    Normal(μ, σ²)       Inv-Gamma(α, β)     (para σ²)       │
└─────────────────────────────────────────────────────────────┘
```

### Ejemplo: Beta-Bernoulli (Clasificación Binaria)

```
Problema: Estimar la probabilidad θ de que un email sea spam
          Observamos n emails, k son spam

Prior:      θ ~ Beta(α, β)
Likelihood: k ~ Binomial(n, θ)
Posterior:  θ|k ~ Beta(α + k, β + n - k)

Visualización del proceso:

Prior Beta(2,2)            Datos: 7/10 spam       Posterior Beta(9,5)
(sin información)                                  (informado)

P(θ)│    ╱──╲           k=7, n=10                P(θ)│      ╱╲
    │   ╱    ╲             spam                      │     ╱  ╲
    │  ╱      ╲                                      │    ╱    ╲
    │ ╱        ╲                                     │   ╱      ╲
    │╱          ╲                                    │  ╱        ╲
    └─────────────── θ                               └─────────────── θ
    0           1                                    0    0.7      1
                                                          ↑
                                                    Pico en ~0.64
```

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def beta_bernoulli_update(alpha_prior: float, beta_prior: float,
                           n_successes: int, n_trials: int):
    """
    Actualización conjugada Beta-Bernoulli.

    Prior: Beta(alpha, beta)
    Likelihood: Binomial(n_trials, theta)
    Posterior: Beta(alpha + n_successes, beta + n_trials - n_successes)
    """
    # Parámetros del posterior
    alpha_post = alpha_prior + n_successes
    beta_post = beta_prior + n_trials - n_successes

    # Crear distribuciones
    prior = stats.beta(alpha_prior, beta_prior)
    posterior = stats.beta(alpha_post, beta_post)

    # Estadísticas
    print(f"Prior: Beta({alpha_prior}, {beta_prior})")
    print(f"  Media: {prior.mean():.3f}")
    print(f"  Varianza: {prior.var():.4f}")

    print(f"\nDatos: {n_successes}/{n_trials} éxitos")

    print(f"\nPosterior: Beta({alpha_post}, {beta_post})")
    print(f"  Media: {posterior.mean():.3f}")
    print(f"  Varianza: {posterior.var():.4f}")

    # Intervalo de credibilidad 95%
    ci_lower, ci_upper = posterior.ppf([0.025, 0.975])
    print(f"  Intervalo credible 95%: [{ci_lower:.3f}, {ci_upper:.3f}]")

    return alpha_post, beta_post

# Ejemplo: Detección de spam
# Prior no muy informativo
alpha_post, beta_post = beta_bernoulli_update(
    alpha_prior=2, beta_prior=2,  # Prior: "no sé mucho"
    n_successes=70,               # 70 emails eran spam
    n_trials=100                  # De 100 revisados
)
```

### Ejemplo: Normal-Normal (Regresión Bayesiana)

```
┌─────────────────────────────────────────────────────────────┐
│  REGRESIÓN LINEAL BAYESIANA                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Modelo:                                                     │
│    y = Xθ + ε,  ε ~ N(0, σ²I)                              │
│                                                              │
│  Prior:                                                      │
│    θ ~ N(μ₀, Σ₀)                                           │
│                                                              │
│  Posterior (forma cerrada):                                  │
│    θ|y ~ N(μₙ, Σₙ)                                         │
│                                                              │
│    Σₙ = (σ⁻²X^TX + Σ₀⁻¹)⁻¹                                │
│    μₙ = Σₙ(σ⁻²X^Ty + Σ₀⁻¹μ₀)                              │
│                                                              │
│  Predicción:                                                 │
│    Para x_nuevo:                                             │
│    y_pred|x ~ N(x^T μₙ, x^T Σₙ x + σ²)                     │
│              ↑           ↑                                   │
│         Media     Incertidumbre                              │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

class BayesianLinearRegression:
    """Regresión lineal bayesiana con prior Normal conjugado."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Args:
            alpha: Precisión del prior (1/varianza)
            beta: Precisión del ruido (1/σ²)
        """
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.mu = None      # Posterior mean
        self.Sigma = None   # Posterior covariance

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calcula el posterior analíticamente."""
        n, d = X.shape

        # Prior: θ ~ N(0, (1/alpha)I)
        # Σ₀⁻¹ = alpha * I
        prior_precision = self.alpha * np.eye(d)

        # Posterior precision
        # Σₙ⁻¹ = β X^T X + Σ₀⁻¹
        posterior_precision = self.beta * X.T @ X + prior_precision

        # Posterior covariance
        self.Sigma = np.linalg.inv(posterior_precision)

        # Posterior mean
        # μₙ = Σₙ (β X^T y + Σ₀⁻¹ μ₀)
        # Con μ₀ = 0
        self.mu = self.beta * self.Sigma @ X.T @ y

        return self

    def predict(self, X_new: np.ndarray, return_std: bool = False):
        """
        Predicción con incertidumbre.

        Returns:
            mean: Predicción media
            std: Desviación estándar (si return_std=True)
        """
        # Media de la predicción
        mean = X_new @ self.mu

        if return_std:
            # Varianza predictiva: x^T Σₙ x + σ²
            # σ² = 1/beta
            var = np.array([
                x @ self.Sigma @ x + 1/self.beta
                for x in X_new
            ])
            std = np.sqrt(var)
            return mean, std

        return mean

    def get_credible_interval(self, X_new: np.ndarray, level: float = 0.95):
        """Calcula intervalo de credibilidad."""
        from scipy import stats

        mean, std = self.predict(X_new, return_std=True)
        z = stats.norm.ppf((1 + level) / 2)

        lower = mean - z * std
        upper = mean + z * std

        return lower, upper


# Ejemplo de uso
np.random.seed(42)
n = 50
X = np.random.randn(n, 3)
theta_real = np.array([1.0, -2.0, 0.5])
y = X @ theta_real + 0.5 * np.random.randn(n)

# Ajustar modelo bayesiano
model = BayesianLinearRegression(alpha=1.0, beta=4.0)  # beta = 1/σ² = 1/0.5² = 4
model.fit(X, y)

print("Parámetros reales:", theta_real)
print("Posterior mean:", model.mu)
print("Posterior std:", np.sqrt(np.diag(model.Sigma)))

# Predicción con incertidumbre
X_test = np.array([[1.0, 0.5, -0.5]])
mean, std = model.predict(X_test, return_std=True)
lower, upper = model.get_credible_interval(X_test)

print(f"\nPredicción: {mean[0]:.3f} ± {std[0]:.3f}")
print(f"Intervalo 95%: [{lower[0]:.3f}, {upper[0]:.3f}]")
```

## 5. MCMC: Cuando No Hay Solución Analítica

### El Problema de la Integral

```
┌─────────────────────────────────────────────────────────────┐
│  LA EVIDENCIA ES INTRATABLE                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Recordar:                                                   │
│                                                              │
│            P(D|θ) · P(θ)                                    │
│  P(θ|D) = ───────────────                                   │
│                P(D)                                          │
│                                                              │
│  Donde:                                                      │
│                                                              │
│    P(D) = ∫ P(D|θ) P(θ) dθ   ← INTEGRAL sobre todo θ       │
│                                                              │
│  Para modelos complejos (redes neuronales, etc):            │
│    • θ tiene millones de dimensiones                        │
│    • La integral es IMPOSIBLE de calcular                   │
│                                                              │
│  Solución: MUESTREAR del posterior sin calcular P(D)        │
│            Eso es MCMC                                       │
└─────────────────────────────────────────────────────────────┘
```

### Idea de MCMC: Random Walk Inteligente

```
┌─────────────────────────────────────────────────────────────┐
│  MARKOV CHAIN MONTE CARLO (MCMC)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Objetivo: Generar muestras θ₁, θ₂, ..., θₙ tales que      │
│            se distribuyen según P(θ|D)                      │
│                                                              │
│  Método: Caminar aleatoriamente por el espacio de θ        │
│          con reglas que aseguran convergencia al posterior  │
│                                                              │
│  Propiedad clave: Solo necesitamos P(D|θ)P(θ)              │
│                   NO necesitamos P(D) (la constante)        │
│                                                              │
│  Resultado: Después de muchas iteraciones ("burn-in"),     │
│             las muestras representan el posterior           │
└─────────────────────────────────────────────────────────────┘
```

### Metropolis-Hastings

```
┌─────────────────────────────────────────────────────────────┐
│  ALGORITMO METROPOLIS-HASTINGS                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ENTRADA: Número de muestras N, estado inicial θ₀          │
│                                                              │
│  Para t = 1 hasta N:                                         │
│    1. PROPONER: θ' ~ q(θ'|θₜ₋₁)                            │
│       (Típicamente: θ' = θₜ₋₁ + ε, ε ~ N(0, σ²))          │
│                                                              │
│    2. CALCULAR ratio de aceptación:                         │
│                                                              │
│           P(θ'|D) q(θₜ₋₁|θ')                               │
│       α = ────────────────────                              │
│           P(θₜ₋₁|D) q(θ'|θₜ₋₁)                             │
│                                                              │
│       Si q es simétrica (random walk):                      │
│           α = P(θ'|D) / P(θₜ₋₁|D)                          │
│             = [P(D|θ')P(θ')] / [P(D|θₜ₋₁)P(θₜ₋₁)]         │
│                                                              │
│    3. ACEPTAR o RECHAZAR:                                    │
│       u ~ Uniforme(0, 1)                                    │
│       Si u < min(1, α):                                     │
│         θₜ = θ'    (aceptar)                               │
│       Sino:                                                  │
│         θₜ = θₜ₋₁  (rechazar, quedarse)                    │
│                                                              │
│  SALIDA: Muestras {θ₁, θ₂, ..., θₙ} (descartar burn-in)   │
└─────────────────────────────────────────────────────────────┘
```

### Visualización de Metropolis-Hastings

```
Espacio de θ con posterior P(θ|D):

    θ₂│
      │      ╱─────╲     Alta probabilidad
      │    ╱         ╲
      │   │     ●───●───●    ← Cadena explorando
      │   │    ╱│         │     zona de alta prob.
      │    ╲  ● │         │
      │     ╲───●─────────│
      │         │     ●───●
      │         │    ╱
      │         ●───●
      └─────────────────────── θ₁

Comportamiento:
  • Propuestas hacia alta prob → aceptadas (α > 1)
  • Propuestas hacia baja prob → a veces rechazadas
  • Después de burn-in → muestras del posterior
```

```python
import numpy as np

def metropolis_hastings(
    log_posterior: callable,
    theta_init: np.ndarray,
    n_samples: int = 10000,
    proposal_std: float = 0.1,
    burn_in: int = 1000
) -> np.ndarray:
    """
    Algoritmo Metropolis-Hastings con propuesta Gaussiana.

    Args:
        log_posterior: Función que calcula log P(D|θ)P(θ)
        theta_init: Estado inicial
        n_samples: Número de muestras a generar
        proposal_std: Desviación estándar de la propuesta
        burn_in: Muestras a descartar al inicio

    Returns:
        samples: Array de muestras del posterior
    """
    d = len(theta_init)
    samples = np.zeros((n_samples + burn_in, d))
    samples[0] = theta_init

    n_accepted = 0
    current_log_prob = log_posterior(theta_init)

    for t in range(1, n_samples + burn_in):
        # 1. Proponer
        theta_current = samples[t-1]
        theta_proposed = theta_current + proposal_std * np.random.randn(d)

        # 2. Calcular log ratio de aceptación
        proposed_log_prob = log_posterior(theta_proposed)
        log_alpha = proposed_log_prob - current_log_prob

        # 3. Aceptar o rechazar
        if np.log(np.random.rand()) < log_alpha:
            samples[t] = theta_proposed
            current_log_prob = proposed_log_prob
            n_accepted += 1
        else:
            samples[t] = theta_current

    acceptance_rate = n_accepted / (n_samples + burn_in)
    print(f"Tasa de aceptación: {acceptance_rate:.2%}")

    # Descartar burn-in
    return samples[burn_in:]


# Ejemplo: Regresión logística bayesiana
def logistic_log_posterior(theta, X, y, prior_std=10.0):
    """Log posterior para regresión logística."""
    # Log-likelihood
    logits = X @ theta
    log_likelihood = np.sum(y * logits - np.log(1 + np.exp(logits)))

    # Log-prior: Normal(0, prior_std²)
    log_prior = -0.5 * np.sum(theta**2) / prior_std**2

    return log_likelihood + log_prior


# Datos sintéticos
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
theta_real = np.array([1.0, -1.5])
prob = 1 / (1 + np.exp(-X @ theta_real))
y = (np.random.rand(n) < prob).astype(float)

# MCMC
log_post = lambda theta: logistic_log_posterior(theta, X, y)
samples = metropolis_hastings(
    log_post,
    theta_init=np.zeros(2),
    n_samples=5000,
    proposal_std=0.1,
    burn_in=1000
)

print(f"\nParámetros reales: {theta_real}")
print(f"Posterior mean: {samples.mean(axis=0)}")
print(f"Posterior std: {samples.std(axis=0)}")
```

### Gibbs Sampling

```
┌─────────────────────────────────────────────────────────────┐
│  GIBBS SAMPLING                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Caso especial de Metropolis-Hastings donde:                │
│    • Muestreamos UNA coordenada a la vez                    │
│    • Usamos la distribución condicional exacta              │
│    • Tasa de aceptación = 100%                              │
│                                                              │
│  Para θ = (θ₁, θ₂, ..., θₐ):                               │
│                                                              │
│  Repetir:                                                    │
│    θ₁ ~ P(θ₁ | θ₂, θ₃, ..., θₐ, D)                        │
│    θ₂ ~ P(θ₂ | θ₁, θ₃, ..., θₐ, D)                        │
│    ...                                                       │
│    θₐ ~ P(θₐ | θ₁, θ₂, ..., θₐ₋₁, D)                      │
│                                                              │
│  Útil cuando las condicionales son fáciles de muestrear    │
│  (ej: modelos con conjugate priors parciales)              │
└─────────────────────────────────────────────────────────────┘
```

```
Comparación visual:

     METROPOLIS-HASTINGS                    GIBBS SAMPLING

θ₂│                                 θ₂│
  │   ╱─╲                             │   ╱─╲
  │  │   │                            │  │   │
  │  │ ● │ ← Movimiento               │  │ ● │───●   ← Primero θ₁
  │  │  ╲╱   en cualquier             │  │   ↓      luego θ₂
  │   ●      dirección                │   ●───●
  │    ╲                              │
  └──────────── θ₁                    └──────────── θ₁

Propuesta diagonal                  Movimientos ortogonales
(todas las coordenadas juntas)      (una coordenada por vez)
```

## 6. Variational Inference: Aproximación Rápida

### El Trade-off MCMC vs VI

```
┌─────────────────────────────────────────────────────────────┐
│  MCMC vs VARIATIONAL INFERENCE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  MCMC:                                                       │
│    ✓ Exacto (asintóticamente)                               │
│    ✓ Funciona para cualquier posterior                      │
│    ✗ LENTO (miles/millones de iteraciones)                  │
│    ✗ Difícil saber si convergió                            │
│                                                              │
│  Variational Inference (VI):                                 │
│    ✓ RÁPIDO (optimización, no muestreo)                    │
│    ✓ Escalable a millones de parámetros                    │
│    ✗ Aproximado (sesgo sistemático)                         │
│    ✗ Puede subestimar incertidumbre                        │
│                                                              │
│  Regla práctica:                                             │
│    • Pocos parámetros + precisión crítica → MCMC           │
│    • Muchos parámetros + velocidad → VI                     │
│    • Deep Learning → VI (ej: VAEs, Bayesian NNs)           │
└─────────────────────────────────────────────────────────────┘
```

### Idea de Variational Inference

```
┌─────────────────────────────────────────────────────────────┐
│  VARIATIONAL INFERENCE                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Objetivo: Aproximar el posterior P(θ|D) con una           │
│            distribución simple q(θ) de una familia Q        │
│                                                              │
│    q*(θ) = argmin KL(q(θ) || P(θ|D))                       │
│              q∈Q                                             │
│                                                              │
│  Problema: KL requiere P(θ|D) que no conocemos             │
│                                                              │
│  Solución: Maximizar el ELBO (Evidence Lower BOund)         │
│                                                              │
│    ELBO(q) = E_q[log P(D,θ)] - E_q[log q(θ)]               │
│            = E_q[log P(D|θ)] - KL(q(θ) || P(θ))            │
│                                                              │
│  Propiedad:                                                  │
│    log P(D) = ELBO(q) + KL(q || P(θ|D))                    │
│             ≥ ELBO(q)   (porque KL ≥ 0)                    │
│                                                              │
│  Maximizar ELBO = Minimizar KL con el posterior            │
└─────────────────────────────────────────────────────────────┘
```

### Mean-Field Variational Inference

```
┌─────────────────────────────────────────────────────────────┐
│  MEAN-FIELD APPROXIMATION                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Restricción: q(θ) se factoriza completamente               │
│                                                              │
│    q(θ) = ∏ᵢ qᵢ(θᵢ)                                        │
│                                                              │
│  Cada qᵢ es independiente (no captura correlaciones)       │
│                                                              │
│  Ejemplo para regresión:                                     │
│    q(θ) = N(μ₁, σ₁²) × N(μ₂, σ₂²) × ... × N(μₐ, σₐ²)     │
│                                                              │
│  Parámetros a optimizar: μ₁, σ₁, μ₂, σ₂, ...              │
│                                                              │
│  Trade-off:                                                  │
│    ✓ Optimización simple y rápida                          │
│    ✗ Subestima incertidumbre (ignora correlaciones)        │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np
from scipy.special import expit  # sigmoid

class VariationalLogisticRegression:
    """
    Regresión logística bayesiana con Variational Inference.
    Aproximación mean-field Gaussiana.
    """

    def __init__(self, prior_std: float = 10.0):
        self.prior_std = prior_std
        self.mu = None      # Posterior mean
        self.log_sigma = None  # Log posterior std (para estabilidad)

    def _elbo(self, X: np.ndarray, y: np.ndarray, n_samples: int = 10):
        """Calcula ELBO con reparametrización."""
        n, d = X.shape

        # Muestrear de q usando reparametrización
        # θ = μ + σ * ε, ε ~ N(0, I)
        sigma = np.exp(self.log_sigma)
        eps = np.random.randn(n_samples, d)
        theta_samples = self.mu + sigma * eps

        # E_q[log P(D|θ)] - estimación Monte Carlo
        log_likelihood = 0
        for theta in theta_samples:
            logits = X @ theta
            ll = np.sum(y * logits - np.log(1 + np.exp(np.clip(logits, -500, 500))))
            log_likelihood += ll
        log_likelihood /= n_samples

        # KL(q || prior) para Gaussianas
        # KL(N(μ,σ²) || N(0,σ_p²)) = 0.5 * (σ²/σ_p² + μ²/σ_p² - 1 - log(σ²/σ_p²))
        prior_var = self.prior_std ** 2
        kl = 0.5 * np.sum(
            sigma**2 / prior_var +
            self.mu**2 / prior_var -
            1 -
            2 * self.log_sigma + np.log(prior_var)
        )

        return log_likelihood - kl

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_epochs: int = 500, lr: float = 0.01):
        """Optimiza ELBO con gradient descent."""
        n, d = X.shape

        # Inicializar
        self.mu = np.zeros(d)
        self.log_sigma = np.zeros(d)

        for epoch in range(n_epochs):
            # Gradiente numérico (simplificado)
            eps = 1e-5
            grad_mu = np.zeros(d)
            grad_log_sigma = np.zeros(d)

            elbo_current = self._elbo(X, y)

            for i in range(d):
                # Gradiente de mu
                self.mu[i] += eps
                elbo_plus = self._elbo(X, y)
                self.mu[i] -= eps
                grad_mu[i] = (elbo_plus - elbo_current) / eps

                # Gradiente de log_sigma
                self.log_sigma[i] += eps
                elbo_plus = self._elbo(X, y)
                self.log_sigma[i] -= eps
                grad_log_sigma[i] = (elbo_plus - elbo_current) / eps

            # Actualizar (ascenso de gradiente porque maximizamos ELBO)
            self.mu += lr * grad_mu
            self.log_sigma += lr * grad_log_sigma

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, ELBO: {elbo_current:.2f}")

        return self

    def predict_proba(self, X: np.ndarray, n_samples: int = 100):
        """Predicción con incertidumbre via Monte Carlo."""
        sigma = np.exp(self.log_sigma)
        probs = []

        for _ in range(n_samples):
            theta = self.mu + sigma * np.random.randn(len(self.mu))
            logits = X @ theta
            probs.append(expit(logits))

        probs = np.array(probs)
        mean = probs.mean(axis=0)
        std = probs.std(axis=0)

        return mean, std
```

## 7. Bayesian Neural Networks (BNNs)

### De Punto a Distribución en Pesos

```
┌─────────────────────────────────────────────────────────────┐
│  RED NEURONAL CLÁSICA vs BAYESIANA                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CLÁSICA:                                                    │
│    • Pesos W son VALORES FIJOS                              │
│    • Una predicción por input                               │
│    • Sin medida de incertidumbre                            │
│                                                              │
│  BAYESIANA:                                                  │
│    • Pesos W son DISTRIBUCIONES P(W|D)                     │
│    • Muchas predicciones (una por muestra de W)            │
│    • Incertidumbre = varianza de predicciones              │
│                                                              │
│  Predicción BNN:                                             │
│    P(y|x, D) = ∫ P(y|x, W) P(W|D) dW                       │
│              ≈ (1/N) Σ P(y|x, Wᵢ),  Wᵢ ~ P(W|D)           │
└─────────────────────────────────────────────────────────────┘
```

### Visualización de Incertidumbre BNN

```
                   RED CLÁSICA                RED BAYESIANA

    y│                                  y│
     │     ●●●●●●●●●●●                   │     ●●●●╱╲●●●●●  ← Zona de alta
     │ ●●●●           ●●●●               │ ●●●●╱    ╲●●●●     incertidumbre
     │                     ●             │     ╱      ╲         (pocos datos)
     │                      ●            │    ╱        ╲
     │                       ?           │              ╲?±
     └───────────────────────── x        └─────────────────── x

    Una línea fija                      Familia de líneas
    "¿Qué pasa aquí?" → No sé         "¿Qué pasa aquí?" →
                                        Predicción ± incertidumbre
```

### Dropout como Aproximación Bayesiana

```
┌─────────────────────────────────────────────────────────────┐
│  MC DROPOUT: Aproximación práctica a BNNs                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Descubrimiento (Gal & Ghahramani, 2016):                   │
│    Usar dropout en INFERENCIA (no solo entrenamiento)       │
│    equivale a muestrear de un posterior aproximado          │
│                                                              │
│  Procedimiento:                                              │
│    1. Entrenar red con dropout normal                       │
│    2. En inferencia, MANTENER dropout activo               │
│    3. Hacer N forward passes con diferentes máscaras       │
│    4. Incertidumbre = varianza de las N predicciones       │
│                                                              │
│  Ventaja: Cualquier red con dropout ya es "bayesiana"      │
│  Costo: N forward passes en vez de 1                        │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

class MCDropoutPredictor:
    """
    Predicción con incertidumbre usando MC Dropout.
    Simula una red neuronal con dropout.
    """

    def __init__(self, weights: np.ndarray, dropout_rate: float = 0.1):
        self.weights = weights
        self.dropout_rate = dropout_rate

    def _forward_with_dropout(self, X: np.ndarray) -> np.ndarray:
        """Forward pass con dropout aleatorio."""
        # Simular dropout: apagar neuronas aleatoriamente
        mask = np.random.binomial(1, 1 - self.dropout_rate, self.weights.shape)
        masked_weights = self.weights * mask / (1 - self.dropout_rate)

        return X @ masked_weights

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100):
        """
        Predicción con estimación de incertidumbre.

        Returns:
            mean: Predicción media
            std: Incertidumbre epistémica (varianza entre muestras)
        """
        predictions = np.array([
            self._forward_with_dropout(X)
            for _ in range(n_samples)
        ])

        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        return mean, std


# Ejemplo
np.random.seed(42)
# Simular pesos de una red entrenada
weights = np.random.randn(10, 1)

predictor = MCDropoutPredictor(weights, dropout_rate=0.2)

# Datos de test
X_test = np.random.randn(5, 10)

mean, uncertainty = predictor.predict_with_uncertainty(X_test, n_samples=100)

for i in range(5):
    print(f"Input {i}: Pred = {mean[i,0]:.3f} ± {uncertainty[i,0]:.3f}")
```

## 8. Aplicaciones Prácticas en ML

### Optimización Bayesiana de Hiperparámetros

```
┌─────────────────────────────────────────────────────────────┐
│  BAYESIAN OPTIMIZATION                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Problema: Encontrar los mejores hiperparámetros           │
│            (learning rate, regularización, etc.)            │
│                                                              │
│  Desafío: Cada evaluación es CARA (entrenar modelo)        │
│                                                              │
│  Solución Bayesiana:                                         │
│    1. Construir modelo Gaussiano (GP) de f(hyperparams)    │
│    2. El GP da predicción + incertidumbre                  │
│    3. Función de adquisición balancea explore/exploit      │
│    4. Evaluar donde la adquisición es máxima               │
│    5. Actualizar GP con nuevo dato, repetir                │
│                                                              │
│  Métrica típica: Expected Improvement (EI)                  │
│    EI(x) = E[max(f(x) - f*, 0)]                            │
│    Alto EI → probable mejora sobre el mejor actual         │
└─────────────────────────────────────────────────────────────┘
```

```
Visualización de Bayesian Optimization:

    f(x)│                    ● Observaciones
        │      ?    ╱╲      ╱╲
        │  ╱───────╱  ╲    ╱  ╲   ← Posterior GP (media)
        │ ╱       ●    ╲──╱    ╲─────●
        │╱                              ╲
        │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← Incertidumbre
        └────────────────────────────────── x
             ↑
        Próximo punto a evaluar
        (alta incertidumbre + potencial mejora)
```

### Cuantificación de Incertidumbre en Diagnóstico

```
┌─────────────────────────────────────────────────────────────┐
│  CASO: Diagnóstico Médico con Incertidumbre                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Modelo clásico:                                             │
│    "Este paciente tiene 87% probabilidad de diabetes"       │
│    → El doctor actúa basándose en 87%                       │
│                                                              │
│  Modelo bayesiano:                                           │
│    "Este paciente tiene 87% ± 15% probabilidad"             │
│    → Alta incertidumbre: pedir más tests                    │
│                                                              │
│    "Este otro tiene 87% ± 3% probabilidad"                  │
│    → Baja incertidumbre: diagnóstico confiable             │
│                                                              │
│  La incertidumbre CAMBIA la decisión clínica               │
└─────────────────────────────────────────────────────────────┘
```

## 9. Resumen: Flujo de Trabajo Bayesiano

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FLUJO DE TRABAJO BAYESIANO EN ML                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐                                                    │
│  │   1. DEFINIR    │                                                    │
│  │     MODELO      │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │   2. ESPECIFICAR│  ← ¿Qué sé antes de ver datos?                    │
│  │     PRIOR       │    Prior informativo vs no informativo            │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │   3. CALCULAR   │  ← ¿Solución cerrada (conjugado)?                 │
│  │    POSTERIOR    │    ¿MCMC (exacto pero lento)?                     │
│  │                 │    ¿VI (rápido pero aproximado)?                  │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  4. POSTERIOR   │  ← Diagnóstico: ¿Convergió MCMC?                  │
│  │   PREDICTIVE    │    ¿ELBO se estabilizó?                           │
│  │    CHECK        │    ¿Predicciones tienen sentido?                  │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  5. PREDICCIÓN  │  ← Media ± incertidumbre                          │
│  │ CON INCERTIDUMBRE│   Intervalos de credibilidad                     │
│  └─────────────────┘                                                    │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CUÁNDO ELEGIR CADA MÉTODO:                                             │
│                                                                          │
│  Conjugate Priors:  Modelos simples (regresión, clasificación básica) │
│  MCMC:              Precisión crítica, pocos parámetros, modelos ricos │
│  VI:                Escalabilidad, deep learning, modelos grandes       │
│  MC Dropout:        Redes existentes, sin reentrenamiento              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Conceptos Clave

```
┌─────────────────────────────────────────────────────────────┐
│  RESUMEN DE CONCEPTOS                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TEOREMA DE BAYES                                            │
│    Posterior ∝ Likelihood × Prior                           │
│    P(θ|D) = P(D|θ)P(θ) / P(D)                              │
│                                                              │
│  PRIOR                                                       │
│    Conocimiento antes de ver datos                          │
│    Informativo vs no informativo                            │
│    Equivale a regularización (L2 = Normal, L1 = Laplace)   │
│                                                              │
│  POSTERIOR                                                   │
│    Distribución de parámetros después de ver datos         │
│    Da incertidumbre, no solo punto estimado                │
│                                                              │
│  CONJUGATE PRIORS                                            │
│    Prior + Likelihood → Posterior de misma familia         │
│    Solución analítica, muy eficiente                       │
│                                                              │
│  MCMC                                                        │
│    Muestrear del posterior cuando no hay solución cerrada  │
│    Metropolis-Hastings, Gibbs Sampling                      │
│    Exacto pero lento                                        │
│                                                              │
│  VARIATIONAL INFERENCE                                       │
│    Aproximar posterior con distribución simple             │
│    Optimizar ELBO en vez de muestrear                      │
│    Rápido pero aproximado                                   │
│                                                              │
│  BNNs                                                        │
│    Distribución sobre pesos de red neuronal                │
│    Incertidumbre predictiva natural                        │
│    MC Dropout como aproximación práctica                   │
└─────────────────────────────────────────────────────────────┘
```

---

**Conexiones con otros temas:**
- Prior Normal → Regularización Ridge (L2)
- Prior Laplace → Regularización Lasso (L1)
- VI → VAEs (Variational Autoencoders)
- BNNs → Active Learning, Optimización Bayesiana
- MCMC → Modelos de lenguaje probabilísticos

**Siguiente:** Teoría de la Información (entropía, KL divergence, conexión con loss functions)
