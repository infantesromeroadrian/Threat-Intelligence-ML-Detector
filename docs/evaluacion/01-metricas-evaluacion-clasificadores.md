# Métricas de Evaluación para Clasificadores

## 1. ¿Por qué Accuracy No Es Suficiente?

### El Problema del Desbalance de Clases

```
Escenario: Detector de fraude bancario

Dataset: 10,000 transacciones
  • 9,900 legítimas (99%)
  • 100 fraudulentas (1%)

Modelo "tonto": Predice SIEMPRE "legítimo"
  Accuracy = 9,900 / 10,000 = 99% 🎉

¿99% accuracy es bueno?
  → NO detecta NINGÚN fraude
  → Es completamente inútil
  → El accuracy miente
```

### Necesitamos Métricas Más Informativas

```
┌────────────────────────────────────────────────────────┐
│  MÉTRICAS DE CLASIFICACIÓN                             │
├────────────────────────────────────────────────────────┤
│                                                        │
│  • Accuracy: % de predicciones correctas (limitada)   │
│  • Precision: ¿Cuántos positivos predichos son reales?│
│  • Recall: ¿Cuántos positivos reales detectamos?      │
│  • F1-Score: Balance entre Precision y Recall         │
│  • Matriz de Confusión: Desglose completo de errores  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 2. Matriz de Confusión

### Estructura

```
                        PREDICCIÓN
                    │  Negativo  │  Positivo
          ──────────┼────────────┼────────────
REAL      Negativo  │    TN      │    FP
          ──────────┼────────────┼────────────
          Positivo  │    FN      │    TP


TN = True Negative   (Correctamente rechazado)
FP = False Positive  (Falsa alarma)
FN = False Negative  (Fallo de detección)
TP = True Positive   (Correctamente detectado)
```

### Ejemplo: Detector de SPAM

```
                          PREDICCIÓN
                      │    HAM    │   SPAM
            ──────────┼───────────┼───────────
 REAL       HAM       │    960    │     6
            ──────────┼───────────┼───────────
            SPAM      │     29    │   120


Interpretación:
  TN = 960: Emails HAM correctamente clasificados como HAM
  FP = 6:   Emails HAM incorrectamente clasificados como SPAM
  FN = 29:  Emails SPAM que se escaparon (clasificados como HAM)
  TP = 120: Emails SPAM correctamente detectados
```

### Visualización Gráfica

```
┌─────────────────────────────────────────────────────────┐
│                    PREDICCIÓN                           │
│              HAM                SPAM                    │
│         ┌──────────────┬──────────────┐                │
│    HAM  │     TN       │     FP       │                │
│         │    960       │      6       │  ← OK (pocos)  │
│ R       │  ✓ Correcto  │  ✗ Falsa     │                │
│ E       │              │    alarma    │                │
│ A       ├──────────────┼──────────────┤                │
│ L       │     FN       │     TP       │                │
│    SPAM │     29       │    120       │                │
│         │  ✗ PELIGRO   │  ✓ Correcto  │                │
│         │   Se escapó  │              │                │
│         └──────────────┴──────────────┘                │
│                                                         │
│  FN (False Negative) es el más peligroso en seguridad: │
│  SPAM/Malware/Ataque que NO detectamos                  │
└─────────────────────────────────────────────────────────┘
```

## 3. Accuracy

### Fórmula

```
              TP + TN
Accuracy = ─────────────────
           TP + TN + FP + FN


              Predicciones correctas
         = ─────────────────────────
             Total de predicciones
```

### Cálculo con Ejemplo

```
             120 + 960        1080
Accuracy = ───────────── = ────────── = 0.9686 (96.86%)
           120+960+6+29       1115

Interpretación:
  El 96.86% de las predicciones son correctas.
  Pero esto NO nos dice cómo se distribuyen los errores.
```

## 4. Precision (Precisión)

### Definición

**Precision:** De todos los que predije como POSITIVO, ¿cuántos realmente lo son?

```
                  TP
Precision = ─────────────
              TP + FP

            Verdaderos Positivos
          = ────────────────────────
            Total predichos Positivos
```

### Cálculo e Interpretación

```
                120
Precision = ───────── = 0.952 (95.2%)
             120 + 6

Interpretación:
  De todos los emails que el modelo marcó como SPAM,
  el 95.2% realmente ERAN spam.

  Solo el 4.8% fueron falsas alarmas (emails legítimos
  marcados como SPAM).
```

### Cuándo Importa Precision

```
┌────────────────────────────────────────────────────────┐
│  PRECISION ES CRÍTICA CUANDO:                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│  • Falsos Positivos son COSTOSOS                       │
│                                                        │
│  Ejemplos:                                             │
│    • Email legítimo → carpeta SPAM (usuario molesto)   │
│    • Usuario legítimo → bloqueado (pérdida de cliente) │
│    • Transacción legal → rechazada (pérdida de venta) │
│                                                        │
│  Alta Precision = Pocas falsas alarmas                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 5. Recall (Sensibilidad / Exhaustividad)

### Definición

**Recall:** De todos los POSITIVOS reales, ¿cuántos detecté?

```
               TP
Recall = ─────────────
           TP + FN

         Verdaderos Positivos
       = ────────────────────────
         Total realmente Positivos
```

### Cálculo e Interpretación

```
              120
Recall = ───────── = 0.805 (80.5%)
          120 + 29

Interpretación:
  De todos los emails que REALMENTE eran SPAM,
  el modelo detectó el 80.5%.

  El 19.5% de los SPAM se escaparon y llegaron
  a la bandeja de entrada.
```

### Cuándo Importa Recall

```
┌────────────────────────────────────────────────────────┐
│  RECALL ES CRÍTICO CUANDO:                             │
├────────────────────────────────────────────────────────┤
│                                                        │
│  • Falsos Negativos son PELIGROSOS                     │
│                                                        │
│  Ejemplos:                                             │
│    • Malware → clasificado como benigno (infección)   │
│    • Ataque → no detectado (brecha de seguridad)      │
│    • Cáncer → no diagnosticado (riesgo vital)         │
│    • Fraude → no detectado (pérdida económica)        │
│                                                        │
│  Alto Recall = Pocos casos peligrosos sin detectar    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 6. El Trade-off Precision vs Recall

### No Puedes Maximizar Ambos

```
         Precision
            │
        1.0 │●
            │ ╲
        0.8 │  ╲
            │   ╲         La curva muestra que
        0.6 │    ╲        al aumentar uno,
            │     ╲       el otro disminuye
        0.4 │      ╲
            │       ╲
        0.2 │        ╲
            │         ●
        0.0 └──────────────── Recall
            0  0.2 0.4 0.6 0.8 1.0


┌─────────────────────────────────────────────────────────┐
│  TRADE-OFF                                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Umbral ALTO (ej: 0.9):                                │
│    • Solo predigo SPAM si estoy MUY seguro             │
│    • Precision ALTA (pocos falsos positivos)           │
│    • Recall BAJO (muchos SPAM se escapan)              │
│                                                         │
│  Umbral BAJO (ej: 0.3):                                │
│    • Predigo SPAM ante cualquier sospecha              │
│    • Recall ALTO (capturo casi todo el SPAM)           │
│    • Precision BAJA (muchas falsas alarmas)            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Ejemplo con Diferentes Umbrales

```
Umbral = 0.9 (muy conservador):
  "Solo marco como SPAM si P(spam) > 0.9"

  Precision: 98%  (casi no hay falsas alarmas)
  Recall:    45%  (más de la mitad del SPAM se escapa)


Umbral = 0.5 (estándar):
  "Marco como SPAM si P(spam) > 0.5"

  Precision: 95%
  Recall:    80%


Umbral = 0.2 (agresivo):
  "Marco como SPAM ante cualquier sospecha"

  Precision: 65%  (muchas falsas alarmas)
  Recall:    98%  (casi no se escapa nada)
```

## 7. F1-Score: El Balance

### Definición

**F1-Score:** Media armónica de Precision y Recall.

```
              2 × Precision × Recall
F1-Score = ────────────────────────────
              Precision + Recall


¿Por qué media ARMÓNICA y no aritmética?
  → Penaliza más cuando uno de los dos es muy bajo
  → Solo es alto si AMBOS son altos
```

### Comparación de Medias

```
Escenario: Precision = 0.95, Recall = 0.10

Media aritmética: (0.95 + 0.10) / 2 = 0.525
  → Parece "decente" pero el Recall es terrible

Media armónica (F1): 2×0.95×0.10 / (0.95+0.10) = 0.181
  → Refleja que el modelo es malo en Recall
```

### Cálculo con Ejemplo

```
Precision = 0.952
Recall = 0.805

           2 × 0.952 × 0.805
F1-Score = ─────────────────── = 0.872 (87.2%)
            0.952 + 0.805

Interpretación:
  F1 = 87.2% indica un buen balance entre
  detectar SPAM (Recall) y no generar
  falsas alarmas (Precision).
```

### Variantes de F-Score

```
┌────────────────────────────────────────────────────────┐
│  F-BETA SCORE                                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│  F_β = (1 + β²) × (Precision × Recall)                │
│        ────────────────────────────────                │
│        (β² × Precision) + Recall                       │
│                                                        │
│  β = 1:   F1 (balance igual)                          │
│  β = 0.5: F0.5 (prioriza Precision)                   │
│  β = 2:   F2 (prioriza Recall)                        │
│                                                        │
│  En SEGURIDAD: F2 suele ser mejor                     │
│  (preferimos detectar todo aunque haya falsas alarmas)│
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 8. Classification Report Completo

### Formato Estándar

```
              precision    recall  f1-score   support

         ham       0.97      1.00      0.98       966
        spam       0.95      0.81      0.87       149

    accuracy                           0.97      1115
   macro avg       0.96      0.90      0.93      1115
weighted avg       0.97      0.97      0.97      1115
```

### Interpretación Línea por Línea

```
┌─────────────────────────────────────────────────────────┐
│  DESGLOSE DEL CLASSIFICATION REPORT                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ham:                                                   │
│    precision=0.97: 97% de los predichos HAM son HAM    │
│    recall=1.00: Detectamos 100% de los HAM reales      │
│    f1-score=0.98: Excelente balance para HAM           │
│    support=966: Había 966 emails HAM en test           │
│                                                         │
│  spam:                                                  │
│    precision=0.95: 95% de los predichos SPAM son SPAM  │
│    recall=0.81: Solo detectamos 81% del SPAM real      │
│    f1-score=0.87: Buen balance pero Recall mejorable   │
│    support=149: Había 149 emails SPAM en test          │
│                                                         │
│  macro avg: Promedio simple de cada métrica            │
│  weighted avg: Promedio ponderado por support          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 9. Cuándo Usar Cada Métrica

### Guía de Decisión

```
┌─────────────────┬──────────────────────────────────────┐
│    MÉTRICA      │           USAR CUANDO                │
├─────────────────┼──────────────────────────────────────┤
│                 │                                      │
│   ACCURACY      │  • Clases balanceadas               │
│                 │  • Todos los errores igual de malos │
│                 │                                      │
├─────────────────┼──────────────────────────────────────┤
│                 │                                      │
│   PRECISION     │  • FP son costosos                  │
│                 │  • Bloquear usuario legítimo = malo │
│                 │  • Email importante → SPAM = malo   │
│                 │                                      │
├─────────────────┼──────────────────────────────────────┤
│                 │                                      │
│   RECALL        │  • FN son peligrosos                │
│                 │  • Malware no detectado = desastre  │
│                 │  • Ataque sin alertar = crítico     │
│                 │                                      │
├─────────────────┼──────────────────────────────────────┤
│                 │                                      │
│   F1-SCORE      │  • Necesitas balance                │
│                 │  • Clases desbalanceadas            │
│                 │  • Comparar modelos                 │
│                 │                                      │
├─────────────────┼──────────────────────────────────────┤
│                 │                                      │
│   F2-SCORE      │  • Recall más importante            │
│                 │  • Seguridad, medicina              │
│                 │  • Mejor sobre-alertar que perder   │
│                 │                                      │
└─────────────────┴──────────────────────────────────────┘
```

### Ejemplos por Dominio

```
DETECTOR DE MALWARE:
  Prioridad: Recall (F2)
  Razón: Un malware no detectado puede infectar la red
  Falso positivo = análisis extra (molesto pero seguro)
  Falso negativo = infección (desastre)


FILTRO DE SPAM:
  Prioridad: Balance (F1) o ligera prioridad a Precision
  Razón: Email importante en SPAM = usuario enfadado
  Mejor dejar pasar algo de SPAM que perder emails


DETECTOR DE FRAUDE:
  Prioridad: Recall (F2)
  Razón: Fraude no detectado = pérdida económica
  Transacción legítima bloqueada = llamada al banco


SISTEMA DE ALERTAS DE SEGURIDAD:
  Prioridad: Recall, pero vigilar Precision
  Razón: Muchas falsas alarmas = "fatiga de alertas"
  El equipo ignora alertas si son siempre falsas
```

## 10. Código Python

### Calcular Todas las Métricas

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Después de hacer predicciones
y_pred = model.predict(X_test)

# Métricas individuales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusión:")
print(cm)

# Reporte completo
print(classification_report(y_test, y_pred,
                           target_names=['ham', 'spam']))
```

### Visualizar Matriz de Confusión

```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['HAM', 'SPAM'],
            yticklabels=['HAM', 'SPAM'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
```

## 11. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│  MÉTRICAS DE CLASIFICACIÓN - RESUMEN                          │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  MATRIZ DE CONFUSIÓN:                                         │
│    TP, TN, FP, FN - Desglose de predicciones                 │
│                                                               │
│  ACCURACY = (TP + TN) / Total                                │
│    Limitada en clases desbalanceadas                         │
│                                                               │
│  PRECISION = TP / (TP + FP)                                  │
│    "De lo que predije positivo, ¿cuánto es real?"            │
│    Alta cuando importa evitar falsas alarmas                 │
│                                                               │
│  RECALL = TP / (TP + FN)                                     │
│    "De lo positivo real, ¿cuánto detecté?"                   │
│    Alta cuando importa no perder casos                       │
│                                                               │
│  F1 = 2×P×R / (P+R)                                          │
│    Balance entre Precision y Recall                          │
│    Usar cuando ambos importan                                │
│                                                               │
│  TRADE-OFF:                                                   │
│    Subir umbral → más Precision, menos Recall                │
│    Bajar umbral → más Recall, menos Precision                │
│                                                               │
│  EN CIBERSEGURIDAD:                                           │
│    Generalmente priorizar RECALL (no perder ataques)         │
│    Pero vigilar Precision (evitar fatiga de alertas)         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Persistencia de modelos y MLOps básico
