# Ficha técnica de los datasets

**Práctica 2 — EDA y preparación de datasets** · ESCOM-IPN · Grupo 6CV2 · Elaborada por Jesús (Dominio y Profiling).
Todas las cifras provienen de `notebooks/01_profiling.ipynb`, calculadas sobre los datos crudos.

## 1. Titanic

| Campo | Descripción |
|---|---|
| **Fuente** | Kaggle, *Titanic Dataset* (yasserh/titanic-dataset): `train.csv` de la competencia *Titanic – Machine Learning from Disaster*, basado en registros históricos de pasajeros (Encyclopedia Titanica). |
| **Licencia** | CC0: dominio público (según Kaggle). |
| **Archivo** | `data/raw/titanic.csv` (61 KB). |
| **Dimensiones** | 891 filas × 12 columnas: 1 objetivo, 3 identificadores, 4 numéricas y 4 categóricas. |
| **Unidad de análisis** | Un pasajero del Titanic (abril de 1912). No incluye a la tripulación ni a los 418 pasajeros del `test.csv` de Kaggle. |
| **Variable objetivo** | `Survived`: 0 = no sobrevivió (549; 61.6 %), 1 = sobrevivió (342; 38.4 %). IR = 1.61 > 1.5: desbalance moderado. |
| **Uso en la práctica** | Valores perdidos y variables mixtas en un problema de clasificación binaria. |

### Variables

| Variable | Tipo | Significado | Dominio observado | Nulos |
|---|---|---|---|---|
| `PassengerId` | identificador | Número consecutivo del registro | 1 – 891 | 0 |
| `Pclass` | categórica ordinal | Clase del boleto (indicador socioeconómico) | 1, 2, 3 (55 % en 3.ª) | 0 |
| `Name` | identificador (texto) | Nombre con título (Mr, Mrs, Miss, Master…) | 891 distintos | 0 |
| `Sex` | categórica | Sexo registrado | male (65 %), female | 0 |
| `Age` | numérica | Edad en años; `xx.5` = estimada | 0.42 – 80; mediana 28 | 177 (19.9 %) |
| `SibSp` | numérica discreta | Hermanos o cónyuges a bordo | 0 – 8 | 0 |
| `Parch` | numérica discreta | Padres o hijos a bordo | 0 – 6 | 0 |
| `Ticket` | cuasi-identificador | Número de boleto (compartido por grupos) | 681 distintos | 0 |
| `Fare` | numérica | Tarifa del boleto, en libras de 1912 | 0 – 512.33; mediana 14.45 | 0 |
| `Cabin` | categórica (alta cardinalidad) | Camarote; la letra es la cubierta | 147 distintos | 687 (77.1 %) |
| `Embarked` | categórica | Puerto: C = Cherbourg, Q = Queenstown, S = Southampton | S (72 %) | 2 (0.2 %) |
| `Survived` | objetivo | ¿Sobrevivió? | 0 / 1 | 0 |

### Problemas de calidad

- **Valores perdidos no aleatorios.** `Cabin` 77.1 % (97.6 % en 3.ª clase vs 18.5 % en 1.ª); `Age` 19.9 % (27.7 % en 3.ª vs 6.0 % en 2.ª); `Embarked` 2 filas, ambas con el mismo boleto.
- **Asimetría y outliers.** `Fare`: asimetría 4.79, 116 outliers IQR (13 %, por encima de 65.63) y máximo de 512.33. `SibSp` (3.70) y `Parch` (2.75) son conteos asimétricos; en `Parch` el IQR vale 0, así que la regla IQR no aplica. `Age` es casi simétrica (0.39).
- **Valores especiales.** 15 tarifas en 0 (boletos de cortesía, no errores); 18 edades estimadas (`xx.5`) y 7 bebés con edad fraccionaria; 8 cabinas con formato atípico (`T`, `D`, `F G73`).
- **Semántica de `Fare`.** Es el precio del boleto completo: el 38.6 % de los pasajeros comparte boleto y, en el 99 % de esos boletos, la tarifa se repite para todos.
- **Inconsistencias.** 10 prefijos de `Ticket` se escriben de varias formas (`SC/Paris` vs `SC/PARIS`, `C.A.` vs `CA`). `Sex` y `Embarked` son consistentes.
- **Sin duplicados ni redundancia.** 0 filas duplicadas; ningún par con \|r\| > 0.8 (el máximo es −0.55, `Pclass`–`Fare`); ninguna columna casi constante.

### Riesgos

- **Fuga de información.** `PassengerId`, `Name`, `Ticket` y `Cabin` son identificadores o cuasi-identificadores y se excluyen del modelo. La ausencia de `Cabin` y `Age` se asocia con no sobrevivir incluso dentro de la misma clase (2.ª clase: sobrevive el 81 % con `Cabin` registrada vs el 44 % sin ella), posiblemente porque esos datos se documentaron después del naufragio. Los indicadores de ausencia deben usarse con cautela.
- **Sesgos y equidad.** `Sex` (sobrevive el 74 % de las mujeres vs el 19 % de los hombres) y `Pclass` (63 % en 1.ª vs 24 % en 3.ª) dominan el modelo. Los faltantes se concentran en 3.ª clase, por lo que la imputación agrega más error justo en ese grupo.
- **Objetivo posterior al evento.** `Survived` solo se conoce después del naufragio: el modelo explica patrones históricos, no predice en tiempo real.

### Limitaciones

- Muestra pequeña (891 pasajeros; 268 en el test de 30 %): las métricas varían bastante según la partición.
- Solo incluye a los pasajeros del `train.csv` de Kaggle, sin tripulación.
- Es un contexto histórico de 1912: lo que aprenda el modelo no se traslada a decisiones actuales.
- Las edades estimadas y las tarifas de cortesía introducen un error de medición que no se puede corregir.

<div class="salto"></div>

## 2. Credit Card Fraud

| Campo | Descripción |
|---|---|
| **Fuente** | Kaggle, *Credit Card Fraud Detection* (mlg-ulb/creditcardfraud). Datos de Worldline y del Machine Learning Group de la Université Libre de Bruxelles. Referencia: Dal Pozzolo, Caelen, Johnson y Bontempi (2015), *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE CIDM. |
| **Licencia** | Open Database License (ODbL 1.0), según Kaggle. |
| **Archivo** | `data/raw/creditcard.csv` (≈ 144 MB). No se versiona en GitHub (límite de 100 MB): cada integrante lo descarga. Verificación: 284,807 filas, 31 columnas y 492 fraudes. |
| **Dimensiones** | 284,807 filas × 31 columnas: 30 numéricas y la objetivo. |
| **Unidad de análisis** | Una transacción con tarjeta de crédito de titulares europeos, durante 2 días de septiembre de 2013. |
| **Variable objetivo** | `Class`: 0 = legítima (284,315; 99.827 %), 1 = fraude (492; 0.173 %). IR = 577.9: desbalance extremo. |
| **Uso en la práctica** | Desbalance extremo y evaluación basada en precision, recall y PR-AUC. |

### Variables

| Variable | Tipo | Significado | Dominio observado | Nulos |
|---|---|---|---|---|
| `Time` | numérica (temporal) | Segundos desde la primera transacción del archivo | 0 – 172,792 (48 h); archivo ordenado por `Time` | 0 |
| `V1` … `V28` | numéricas (componentes PCA) | Componentes principales de atributos que no se publican por confidencialidad | media 0; desviación de 1.96 (`V1`) a 0.33 (`V28`) | 0 |
| `Amount` | numérica | Monto de la transacción (la fuente no indica la moneda) | 0 – 25,691.16; mediana 22.00 | 0 |
| `Class` | objetivo | 1 = fraude, 0 = legítima | 0 / 1 | 0 |

### Problemas de calidad

- **Desbalance extremo.** Solo el 0.173 % son fraudes: predecir siempre "no fraude" da 99.83 % de accuracy y 0 % de recall.
- **Duplicados.** 1,081 filas idénticas (0.38 %; 19 fraudes), sin etiquetas contradictorias. Con el split del equipo, 456 filas de test (7 de los 148 fraudes de test) tienen una copia exacta en train. Otras 8,063 filas repiten el vector `V1`–`V28` y el monto de otra fila en otro momento.
- **Asimetría y colas pesadas.** `Amount` tiene asimetría 16.98 (media 88.35 vs mediana 22.00; p99 = 1,017.97; 11.2 % de outliers IQR). La curtosis supera 200 en 8 de las `V` y en `Amount`.
- **Outliers informativos.** El 87 % de los fraudes son outliers IQR en `V14` (vs el 4.8 % de las legítimas) y el 97 % tiene al menos un outlier en alguna `V`: no son errores, son la señal de fraude.
- **Escalas heterogéneas.** `Time` llega a 172,792, `Amount` a 25,691 y las `V` rara vez pasan de ±5: hay que escalar.
- **Valores especiales.** 1,825 montos en 0 (27 de ellos fraudes); son válidos.
- **Sin otros problemas.** No hay nulos, categóricas ni columnas casi constantes (salvo `Class`), y ningún par tiene \|r\| > 0.8: las `V` son ortogonales y el máximo es 0.53 (`V2`–`Amount`).

### Riesgos

- **Fuga temporal.** `Time` es relativo al inicio del archivo, que está ordenado; ese valor no existe en producción. El fraude se concentra en ciertas horas (1.71 % en la hora 2 vs ≈ 0.1 % de día), y el split aleatorio mezcla pasado y futuro.
- **Fuga por duplicados.** Las copias exactas en train y test inflan las métricas, sobre todo el recall.
- **Auditoría imposible.** `V1`–`V28` son anónimas: no se puede verificar si alguna codifica información posterior a la transacción o atributos sensibles (ubicación, tipo de comercio).
- **Costos asimétricos y equidad.** Un falso positivo bloquea a un cliente legítimo y un falso negativo deja pasar un fraude. Los clientes con hábitos poco comunes (por ejemplo, compras de madrugada) recibirían más bloqueos injustificados.
- **Etiquetas con retraso y ruido.** El fraude se confirma días después de la compra, y los fraudes no detectados quedan etiquetados como 0.

### Limitaciones

- Solo cubre 2 días de 2013 y titulares europeos: los patrones de fraude cambian con el tiempo (*concept drift*).
- Hay pocos positivos (492 fraudes; 148 en test), así que las métricas de la clase minoritaria son inestables.
- Sin identificadores de tarjeta o cliente no se puede dividir por cliente ni construir variables de historial.
- La anonimización por PCA impide interpretar el modelo en términos de negocio.

---

*Elaborada por Jesús con apoyo de IA generativa (Claude), según se declara en el README.*
