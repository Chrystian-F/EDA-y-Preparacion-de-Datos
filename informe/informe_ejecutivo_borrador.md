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

**Imputación (Rol D).** Solo Titanic tiene valores perdidos: `Cabin` 77 %, `Age` 20 % y `Embarked` 0.2 %; Fraud no tiene ninguno.
`Age` falta mucho más en 3ª clase (28 %) que en 1ª (14 %) y 2ª (6 %), y quienes no tienen edad sobrevivieron menos (29 % vs 41 %):
la ausencia **no es aleatoria (MAR)**. Se compararon 9 estrategias con el mismo modelo (F1 en test):

| Estrategia | F1 | ROC-AUC |
|---|---|---|
| Eliminar filas con `Age` nula (solo train) | 0.703 | 0.853 |
| Mediana global (línea base) | 0.733 | 0.849 |
| Media | 0.743 | 0.849 |
| KNN (k=5) | 0.734 | 0.849 |
| Iterativa (MICE) | 0.750 | 0.851 |
| Mediana por `Pclass`×`Sex` + indicador de faltante | 0.746 | 0.850 |
| Lo anterior + `tiene_cabina` | 0.755 | 0.847 |
| **Mediana por título (`Mr`/`Mrs`/`Miss`/`Master`) + indicador** | **0.778** | **0.880** |

Se eligió **mediana de `Age` por título del nombre + indicador `Age_faltante`** (sugerencia del Rol B) y `Embarked` con la moda;
`Cabin` se excluye. El título distingue a los niños (`Master`) mejor que `Pclass`×`Sex`, conserva mejor la distribución de `Age`
que la mediana global (que crea un pico artificial en 28 años) y da el mejor resultado. Parte de la mejora se debe a que `Titulo`
también entra como atributo al modelo. Si el equipo no quiere agregar atributos, la alternativa es mediana por `Pclass`×`Sex` + indicador.
Eliminar filas es lo peor: pierde 20 % de train, sobre todo de 3ª clase. Todo se ajusta solo con train (`src/rol_d.py`).

**Outliers (Rol D).** Se detectaron con la regla IQR (Tukey) sobre train y se compararon: no tratar, winsorizar con IQR,
winsorizar con percentiles 1–99, `log1p` y eliminar filas de train.

- **Titanic — `Fare`:** límites IQR [−25.3, 63.2] (12 % de outliers, máximo 512.3); límites P1–P99 [0, 227.5].
  Con P1–P99 la media de train pasa de 30.9 a 29.8 y la desviación estándar de 47.8 a 39.8.
  Los boletos caros son reales (1ª clase) y el 74 % de esos pasajeros sobrevivió. El efecto en métricas es mínimo
  (F1 0.743–0.750); se eligió **`log1p(Fare)`** por dar el mejor ROC-AUC/PR-AUC sin perder información.
  `Parch` y `SibSp` no se tratan: en `Parch` Q1 = Q3 = 0 y la regla IQR marcaría como atípico a cualquiera con familia;
  winsorizar todas las numéricas baja el F1 a 0.732 y eliminar filas a 0.717.
- **Fraud:** 10 de 30 columnas tienen más del 5 % de outliers IQR, y entre ellos el fraude llega a 37 % (V11) frente a 0.17 % global:
  **los outliers son la señal**. Eliminar filas deja 9 de 344 fraudes en train (PR-AUC 0.031). Se eligió **winsorizar P1–P99
  todas las columnas** (p. ej. `Amount` a [0.12, 1015]; desviación estándar 253 → 159): recall 0.615 → 0.750,
  F1 0.714 → 0.801 y PR-AUC 0.708 → 0.773, porque limita la influencia de valores extremos en la regresión logística sin
  descartar ningún fraude. IQR es más agresivo y queda un poco abajo (PR-AUC 0.755).
  El Rol B recomendaba winsorizar solo `Amount`; la prueba muestra que eso no cambia nada (PR-AUC 0.708) y que la mejora viene
  de recortar las `V`. Con P1–P99 el fraude sigue en el extremo de cada variable, así que la señal se conserva.

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

- **Pipeline Titanic** = mediana de `Age` por título + indicador + `log1p(Fare)` + `class_weight='balanced'`:
  precision 0.739, recall 0.825, F1 0.780, ROC-AUC 0.881 (85 de 103 sobrevivientes detectados, 30 falsos positivos).
  Sin balanceo: precision 0.792, recall 0.777, F1 0.784.
- **Pipeline Fraud** = winsorización P1–P99 (sin balanceo): precision 0.860, recall 0.750, F1 0.801, PR-AUC 0.773
  (111 fraudes detectados, 18 falsas alarmas). Al agregar `class_weight='balanced'` el recall sube a 0.878 (130 fraudes)
  pero las falsas alarmas pasan a 2,114 y la PR-AUC baja a 0.750.


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
- La imputación de `Age` asume que pasajeros con el mismo título tienen edades parecidas; si el faltante fuera MNAR
  (depende de la propia edad) ninguna imputación lo corrige. La mediana por grupo reduce la varianza real de `Age`, y la mejora de F1 mezcla el efecto de imputar con el de agregar `Titulo`.
- Los límites de winsorización se calculan con un solo split de train; con otro split cambiarían un poco.
- En Fraud no se puede saber si un valor extremo es un error o una transacción real, porque V1–V28 son anónimas.
