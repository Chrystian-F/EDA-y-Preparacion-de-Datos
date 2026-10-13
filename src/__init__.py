"""
Utilidades comunes de la Practica 2 (EDA y preparación de datasets).

Uso típico desde un notebook en notebooks/:
    import sys; sys.path.append("..")
    from src.utils import *

    df = cargar_datos("titanic")
    X_train, X_test, y_train, y_test = dividir(df, "titanic")
    modelo = pipeline_base(X_train)
    res = evaluar(modelo, X_train, y_train, X_test, y_test, nombre="Línea base")
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Reglas comunes para utilizar en todas las etapas
# ---------------------------------------------------------------------------
RANDOM_STATE = 42      # semilla fija -> resultados reproducibles
TEST_SIZE = 0.30       # Hold-Out 70 % train / 30 % test (visto en clase)

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = RAIZ / "data" / "raw"
RESULTADOS_DIR = RAIZ / "resultados"

# Configuracion de cada dataset.
# "excluir" = identificadores o columnas de texto libre que no se usan como
# atributos en la linea base (riesgo de fuga o sin valor predictivo directo).
DATASETS = {
    "titanic": {
        "archivo": "titanic.csv",
        "target": "Survived",
        "excluir": ["PassengerId", "Name", "Ticket", "Cabin"],
    },
    "fraud": {
        "archivo": "creditcard.csv",
        "target": "Class",
        "excluir": [],
    },
}


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
def cargar_datos(nombre):
    """Devuelve el DataFrame crudo ('titanic' o 'fraud') sin modificar."""
    if nombre not in DATASETS:
        raise ValueError(f"Dataset desconocido: {nombre}. Usa {list(DATASETS)}")
    ruta = DATA_DIR / DATASETS[nombre]["archivo"]
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descárgalo (ver README) y colócalo en data/raw/."
        )
    return pd.read_csv(ruta)


# ---------------------------------------------------------------------------
# 2. Separación train / test (Hold-Out estratificado)
# ---------------------------------------------------------------------------
def dividir(df, nombre, excluir=None):
    """
    Separa X / y y hace el split 70/30 estratificado por la clase.

    excluir: lista de columnas a quitar; si es None usa la de DATASETS.
    Devuelve: X_train, X_test, y_train, y_test
    """
    cfg = DATASETS[nombre]
    excluir = cfg["excluir"] if excluir is None else excluir
    y = df[cfg["target"]]
    X = df.drop(columns=[cfg["target"]] + [c for c in excluir if c in df.columns])
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


# ---------------------------------------------------------------------------
# 3. Preprocesamiento mínimo (solo para que el modelo pueda correr)
# ---------------------------------------------------------------------------
def columnas_por_tipo(X):
    """Devuelve (numéricas, categóricas)."""
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return num, cat


def preprocesador_minimo(X, imputar_num="median", imputar_cat="most_frequent"):
    """
    ColumnTransformer que imputa, escala numéricas y aplica one-hot a categóricas.
    Al ir dentro de un Pipeline se ajusta SOLO con train (sin fuga de información).

    El Rol D puede cambiar las estrategias, por ejemplo:
        preprocesador_minimo(X_train, imputar_num="mean")
        preprocesador_minimo(X_train, imputar_cat="constant")  # -> 'Desconocido'
    """
    num, cat = columnas_por_tipo(X)
    imp_cat = (SimpleImputer(strategy="constant", fill_value="Desconocido")
               if imputar_cat == "constant" else SimpleImputer(strategy=imputar_cat))
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputar", SimpleImputer(strategy=imputar_num)),
            ("escalar", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("imputar", imp_cat),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])


# ---------------------------------------------------------------------------
# 4. Modelo fijo y evaluación
# ---------------------------------------------------------------------------
def modelo_base(**kwargs):
    """Clasificador fijo del equipo. kwargs permite, p. ej., class_weight='balanced'."""
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **kwargs)


def pipeline_base(X_train, **kwargs_modelo):
    """Pipeline listo: preprocesamiento mínimo + modelo fijo."""
    return Pipeline([
        ("prep", preprocesador_minimo(X_train)),
        ("modelo", modelo_base(**kwargs_modelo)),
    ])


def evaluar(modelo, X_train, y_train, X_test, y_test, nombre="", mostrar=True):
    """
    Entrena con train, evalúa en test y devuelve un diccionario de métricas.
    Funciona con modelos o Pipelines de sklearn e imblearn (SMOTE, etc.).
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_score = modelo.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    res = {
        "experimento": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }
    if mostrar:
        print(f"--- {nombre} ---")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            print(f"{k:>10}: {res[k]:.4f}")
        print(f"Matriz de confusión  TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")
    return res


def guardar_resultados(resultados, archivo):
    """Guarda una lista de diccionarios de evaluar() en resultados/<archivo>.csv."""
    RESULTADOS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(resultados)
    df.to_csv(RESULTADOS_DIR / archivo, index=False)
    return df