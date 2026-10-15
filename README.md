# Práctica 2 — EDA y preparación de datasets

**Inteligencia Artificial · ESCOM-IPN · Grupo 6CV2**

Proceso reproducible para diagnosticar la calidad de dos datasets, justificar decisiones de
preprocesamiento (imputación, winsorización y balanceo) y medir su impacto sobre un clasificador.

## Equipo y roles

| Rol | Responsable | Secciones |
|---|---|---|
| A — Integrador | Chrystian | Repositorio, split, utilidades, línea base, notebook final |
| B — Dominio y Profiling | Jesús | Comprensión del dominio, data profiling, ficha técnica |
| C — EDA | Ulises | Análisis exploratorio, presentación |
| D — Imputación y Outliers | Jorge | Imputación de valores perdidos, winsorización |
| E — Desbalance | Damián | Tratamiento del desbalance, informe ejecutivo |

## Datasets

| Dataset | Archivo | Objetivo | Fuente |
|---|---|---|---|
| Titanic | `data/raw/titanic.csv` | `Survived` | [Kaggle — yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| Credit Card Fraud | `data/raw/creditcard.csv` | `Class` | [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (licencia ODbL) |

> **`creditcard.csv` no está en el repositorio** (pesa ~144 MB y GitHub acepta máximo 100 MB).
> Descárgalo de Kaggle (*Download → Download dataset as zip*), descomprímelo y coloca
> `creditcard.csv` en `data/raw/`. No abras ni guardes los CSV con Excel.

## Instalación

```bash
git clone <URL-del-repositorio>
cd EDA-y-Preparacion-de-Datos
python -m venv .venv
source .venv/Scripts/activate      # Git Bash en Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Estructura

```
data/raw/        datasets originales (sin modificar)
notebooks/       un notebook por sección
src/utils.py     funciones comunes: cargar_datos, dividir, preprocesador_minimo, evaluar...
resultados/      métricas de cada experimento (CSV)
figuras/         gráficas exportadas (PNG)
docs/            entregables en documento (ficha técnica, …)
```

| Notebook | Responsable |
|---|---|
| `00_base.ipynb` — línea base | Rol A |
| `01_profiling.ipynb` | Rol B |
| `02_eda.ipynb` | Rol C |
| `03_imputacion_outliers.ipynb` | Rol D |
| `04_balanceo.ipynb` | Rol E |
| `05_final.ipynb` — integración | Rol A |

## Reglas comunes (acordadas por el equipo)

1. Semilla fija: `RANDOM_STATE = 42`.
2. Split Hold-Out estratificado 70 % train / 30 % test (`dividir()`).
3. Clasificador fijo: `LogisticRegression(max_iter=1000)` (`modelo_base()`).
4. Métricas: accuracy, precision, recall, F1, ROC-AUC, PR-AUC y matriz de confusión (`evaluar()`).
5. Toda transformación se ajusta **solo con train**; el balanceo se aplica **solo a train**.
6. Cada quien trabaja en su rama (`rol-b-profiling`, …) y abre un Pull Request hacia `main`.

## Uso rápido

```python
import sys; sys.path.append("..")   # desde notebooks/
from src.utils import *

df = cargar_datos("titanic")
X_train, X_test, y_train, y_test = dividir(df, "titanic")
res = evaluar(pipeline_base(X_train), X_train, y_train, X_test, y_test, nombre="Línea base")
```

## Uso de IA generativa

Cada integrante declara aquí dónde utilizó herramientas de IA generativa:

| Integrante | Herramienta | Uso |
|---|---|---|
| Chrystian | Claude | Estructura del repositorio y `src/utils.py` (revisado y probado por el equipo) |
| Jorge | Claude | Estructura de `src/rol_d.py`, `notebooks/03_imputacion_outliers.ipynb` y redacción de las secciones de imputación y outliers del informe (revisado y ejecutado por Jorge) |
| Jesús | Claude | Borrador de `notebooks/01_profiling.ipynb` (funciones de profiling e interpretaciones) y de la ficha técnica (`docs/ficha_tecnica.md`); las cifras salen de ejecutar el notebook |
| | | |
