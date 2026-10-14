# Informe ejecutivo — Práctica 2 EDA

**Datasets:** Titanic (891 pasajeros) y Credit Card Fraud (284,807 transacciones)
**Modelo fijo:** `LogisticRegression(max_iter=1000)` · split 70/30 estratificado · `random_state=42`

> **Borrador del Rol E.** Las secciones marcadas con ✏️ esperan los resultados de B y D.
> Las cifras de balanceo se generaron con `src/utils.py` del Rol A (imputación mediana/moda, one-hot y escalado).
> Si D cambia la imputación o B decide excluir `Time` en Fraud por posible fuga, hay que volver a correr el notebook.

---

## 1. Hallazgos relevantes

**Calidad de datos** ✏️ *(Rol B: pegar aquí los 3–5 hallazgos principales de la tabla resumen del profiling)*

**Desbalance de clases (Rol E)**

| Dataset | Clase 0 | Clase 1 | % minoritaria | IR | ¿Desbalanceado? |
|---|---|---|---|---|---|
| Titanic | 549 | 342 | 38.38 % | 1.61 | Sí, leve |
| Fraud | 284,315 | 492 | 0.17 % | 577.88 | Sí, extremo |

En Fraud, un modelo que siempre responde "no fraude" obtiene **99.83 % de accuracy** sin detectar un solo fraude.
Por eso el accuracy no sirve para comparar modelos aquí; se usan precision, recall, F1 y PR-AUC de la clase minoritaria.

## 2. Justificación de decisiones

**Imputación** ✏️ *(Rol D: estrategia elegida, columnas, comparación media/moda vs mediana/"Desconocido" y por qué)*

**Outliers (winsorización)** ✏️ *(Rol D: variables, límites exactos con IQR y percentiles 1–99, efecto en estadísticas y métricas)*

**Balanceo (Rol E).** Se compararon 5 estrategias con el mismo modelo, aplicando el remuestreo solo a train:
sin balanceo, `class_weight='balanced'`, RandomOverSampler, RandomUnderSampler y SMOTE.
Se eligió **`class_weight='balanced'` en ambos datasets** porque alcanza el mismo recall que ROS y SMOTE sin duplicar
ni inventar registros, y sin descartar datos como RUS.

## 3. Resultados

**Titanic** (test: 268 pasajeros, 103 sobrevivientes)

| Estrategia | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Sin balanceo | 0.747 | 0.718 | 0.733 | 0.849 |
| **class_weight='balanced'** | **0.716** | **0.806** | **0.758** | **0.849** |
| RandomOverSampler | 0.703 | 0.806 | 0.751 | 0.848 |
| RandomUnderSampler | 0.726 | 0.796 | 0.759 | 0.852 |
| SMOTE | 0.722 | 0.806 | 0.761 | 0.851 |

**Credit Card Fraud** (test: 85,443 transacciones, 148 fraudes)

| Estrategia | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fraudes detectados | Falsas alarmas |
|---|---|---|---|---|---|---|---|
| Sin balanceo | 0.850 | 0.615 | 0.714 | 0.957 | 0.708 | 91 | 16 |
| **class_weight='balanced'** | **0.067** | **0.878** | **0.125** | **0.968** | **0.705** | **130** | **1,807** |
| RandomOverSampler | 0.067 | 0.878 | 0.124 | 0.968 | 0.705 | 130 | 1,817 |
| RandomUnderSampler | 0.064 | 0.885 | 0.119 | 0.971 | 0.586 | 131 | 1,914 |
| SMOTE | 0.065 | 0.878 | 0.120 | 0.967 | 0.706 | 130 | 1,885 |

Gráficas: `figuras/e_metricas_estrategias.png`, `figuras/e_curvas_pr_fraud.png`, `figuras/e_matrices_*.png`.

**Lectura de los resultados**
- En **Titanic** el balanceo sube el recall de 0.72 a ~0.81 con una pérdida pequeña de precision. Las 4 técnicas
  quedan dentro de 1–3 pasajeros entre sí, así que la diferencia no es significativa.
- En **Fraud** el balanceo detecta ~40 fraudes más, pero multiplica las falsas alarmas de 16 a ~1,800–1,900. El PR-AUC
  casi no cambia (0.705–0.708), lo que indica que balancear **no mejora la capacidad de distinguir fraudes**:
  solo desplaza el umbral de decisión. RUS es claramente peor (PR-AUC 0.59) porque se queda con 688 filas de train.

## 4. Recomendación final

✏️ *(Integrar con las decisiones de D: "pipeline = imputación X + winsorización Y + balanceo Z")*

- **Titanic:** usar `class_weight='balanced'`. Mejora el recall sin crear datos artificiales; el desbalance es leve,
  así que la ganancia es moderada.
- **Fraud:** usar `class_weight='balanced'` **si la prioridad es detectar fraudes** (recall 0.88), y ajustar el
  umbral de decisión con una partición de validación tomada de train para reducir las falsas alarmas.
  Si revisar alarmas es costoso, el modelo **sin balanceo** ofrece el mejor equilibrio (F1 0.71).
- No usar RandomUnderSampler en Fraud: pierde casi toda la información de la clase mayoritaria.

## 5. Limitaciones

- Un solo split 70/30: en Titanic el test es pequeño (268 filas) y las diferencias entre estrategias caben en el ruido.
  Validación cruzada estratificada daría estimaciones más estables.
- El umbral de decisión se dejó en 0.5 para todas las estrategias; no se optimizó.
- Un solo modelo (regresión logística, por regla del equipo); otros modelos podrían responder distinto al balanceo.
- En Fraud las variables V1–V28 son componentes PCA anónimos, lo que impide interpretar qué rasgos indican fraude.
- Titanic contiene variables sensibles (sexo, edad, clase): el modelo puede reproducir desigualdades históricas. ✏️ *(Rol B: completar con el análisis de sesgo)*
- ✏️ *(Rol D: limitaciones de la imputación y la winsorización)*
