"""
Rol D — Imputación de valores perdidos y tratamiento de outliers.

Transformadores compatibles con sklearn para que el Rol A los integre en el
notebook final (05_final.ipynb). TODOS se ajustan con fit() SOLO sobre train
al ir dentro de un Pipeline, por lo que no hay fuga de información hacia test.

Uso típico:
    from src.utils import *
    from src.rol_d import preprocesador_d, pipeline_d

    modelo = pipeline_d(X_train, imputar_num="grupo", outliers="iqr")
    res = evaluar(modelo, X_train, y_train, X_test, y_test, nombre="D: grupo + IQR")
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   RobustScaler, StandardScaler)

from src.utils import RANDOM_STATE, columnas_por_tipo, modelo_base


# ---------------------------------------------------------------------------
# 1. Diagnóstico
# ---------------------------------------------------------------------------
def tabla_faltantes(df):
    """Conteo y porcentaje de valores perdidos por columna (solo las que tienen)."""
    n = df.isna().sum()
    t = pd.DataFrame({"faltantes": n, "%": 100 * n / len(df), "tipo": df.dtypes.astype(str)})
    return t[t["faltantes"] > 0].sort_values("%", ascending=False)


def limites_iqr(s, k=1.5):
    """Límites de Tukey: [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def tabla_outliers(X, y=None, k=1.5, columnas=None):
    """
    Cuenta outliers por la regla IQR en cada columna numérica.
    Si se da y, reporta qué % de los outliers pertenece a la clase positiva
    (clave en fraude: muchos fraudes SON outliers).
    """
    columnas = columnas or columnas_por_tipo(X)[0]
    filas = []
    for c in columnas:
        s = X[c].dropna()
        if s.nunique() <= 2:          # binarias: no aplica
            continue
        li, ls = limites_iqr(s, k)
        mask = (X[c] < li) | (X[c] > ls)
        fila = {"columna": c, "lim_inf": li, "lim_sup": ls,
                "n_outliers": int(mask.sum()), "%_outliers": 100 * mask.mean(),
                "min": s.min(), "max": s.max(), "asimetria": s.skew()}
        if y is not None:
            yy = np.asarray(y)[mask.to_numpy()]
            fila["%_clase1_en_outliers"] = 100 * yy.mean() if len(yy) else np.nan
            fila["%_clase1_global"] = 100 * np.mean(y)
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("%_outliers", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Transformadores
# ---------------------------------------------------------------------------
class ImputadorPorGrupo(BaseEstimator, TransformerMixin):
    """
    Imputa una columna numérica con la mediana de su grupo (p. ej. Age por
    Pclass y Sex), aprendida SOLO con train. Si un grupo no se vio en train,
    usa la mediana global de train. Trabaja sobre DataFrames.
    """

    def __init__(self, columna="Age", grupos=("Pclass", "Sex")):
        self.columna = columna
        self.grupos = grupos

    def fit(self, X, y=None):
        g = list(self.grupos)
        self.medianas_ = X.groupby(g)[self.columna].median()
        self.global_ = X[self.columna].median()
        return self

    def transform(self, X):
        X = X.copy()
        g = list(self.grupos)
        llaves = pd.MultiIndex.from_frame(X[g]) if len(g) > 1 else X[g[0]]
        relleno = pd.Series(self.medianas_.reindex(llaves).to_numpy(), index=X.index)
        relleno = relleno.fillna(self.global_)
        X[self.columna] = X[self.columna].fillna(relleno)
        return X


class Winsorizador(BaseEstimator, TransformerMixin):
    """
    Recorta (clip) cada columna a límites aprendidos en train.
      metodo="iqr"        -> [Q1 - k·IQR, Q3 + k·IQR]   (Tukey)
      metodo="percentil"  -> [p_inf, p_sup]  (por defecto 1 % y 99 %)
    No elimina filas: conserva el tamaño de train y test.
    """

    def __init__(self, metodo="iqr", k=1.5, p_inf=0.01, p_sup=0.99):
        self.metodo = metodo
        self.k = k
        self.p_inf = p_inf
        self.p_sup = p_sup

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.metodo == "iqr":
            q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
            iqr = q3 - q1
            self.inf_, self.sup_ = q1 - self.k * iqr, q3 + self.k * iqr
        elif self.metodo == "percentil":
            self.inf_, self.sup_ = np.nanpercentile(X, [100 * self.p_inf, 100 * self.p_sup], axis=0)
        else:
            raise ValueError("metodo debe ser 'iqr' o 'percentil'")
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.inf_, self.sup_)

    def get_feature_names_out(self, input_features=None):
        return input_features


def _log1p_seguro(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


# ---------------------------------------------------------------------------
# 3. Preprocesador configurable del Rol D
# ---------------------------------------------------------------------------
IMPUTADORES_NUM = ("mean", "median", "grupo", "knn", "iterativo")
METODOS_OUTLIERS = (None, "iqr", "percentil", "log")


def _imputador_num(estrategia):
    if estrategia in ("mean", "median"):
        return SimpleImputer(strategy=estrategia)
    if estrategia == "grupo":            # el grueso lo hace ImputadorPorGrupo antes;
        return SimpleImputer(strategy="median")   # esto cubre otras columnas numéricas
    if estrategia == "knn":
        return KNNImputer(n_neighbors=5)
    if estrategia == "iterativo":
        return IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
    raise ValueError(f"imputar_num debe ser uno de {IMPUTADORES_NUM}")


def preprocesador_d(X, imputar_num="median", imputar_cat="most_frequent",
                    indicador=False, outliers=None, cols_outliers=None,
                    escalador="standard", k=1.5, p_inf=0.01, p_sup=0.99,
                    grupo_col="Age", grupos=("Pclass", "Sex")):
    """
    Devuelve un Pipeline (DataFrame -> matriz numérica) con:
      1) imputación por grupo opcional (imputar_num="grupo"),
      2) imputación numérica (mean/median/knn/iterativo) [+ indicador de faltante],
      3) tratamiento de outliers en cols_outliers (None/'iqr'/'percentil'/'log'),
      4) escalado ('standard' o 'robust') y one-hot de categóricas.

    Para KNN e iterativo se escala ANTES de imputar (las distancias dependen de la escala).
    """
    num, cat = columnas_por_tipo(X)
    if outliers not in METODOS_OUTLIERS:
        raise ValueError(f"outliers debe ser uno de {METODOS_OUTLIERS}")
    cols_out = [c for c in (cols_outliers if cols_outliers is not None else num) if c in num]
    resto = [c for c in num if c not in cols_out] if outliers else num
    if not outliers:
        cols_out = []

    Esc = RobustScaler if escalador == "robust" else StandardScaler
    imp_cat = (SimpleImputer(strategy="constant", fill_value="Desconocido")
               if imputar_cat == "constant" else SimpleImputer(strategy=imputar_cat))

    def rama_num(cols, tratamiento):
        pasos = []
        if imputar_num in ("knn", "iterativo"):
            pasos += [("pre_escala", StandardScaler())]
        pasos += [("imputar", _imputador_num(imputar_num))]
        if tratamiento in ("iqr", "percentil"):
            pasos += [("winsorizar", Winsorizador(tratamiento, k=k, p_inf=p_inf, p_sup=p_sup))]
        elif tratamiento == "log":
            pasos += [("log1p", FunctionTransformer(_log1p_seguro, feature_names_out="one-to-one"))]
        pasos += [("escalar", Esc())]
        return Pipeline(pasos)

    transformadores = []
    if resto:
        transformadores.append(("num", rama_num(resto, None), resto))
    if cols_out:
        transformadores.append(("num_out", rama_num(cols_out, outliers), cols_out))
    if indicador:
        cols_nan = [c for c in num if X[c].isna().any()]
        if cols_nan:
            transformadores.append(("faltante", MissingIndicator(features="all"), cols_nan))
    if cat:
        transformadores.append(("cat", Pipeline([
            ("imputar", imp_cat),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat))

    pasos = []
    if imputar_num == "grupo" and grupo_col in X.columns and all(g in X.columns for g in grupos):
        pasos.append(("imputar_grupo", ImputadorPorGrupo(grupo_col, grupos)))
    pasos.append(("columnas", ColumnTransformer(transformadores)))
    return Pipeline(pasos)


def pipeline_d(X_train, modelo_kwargs=None, **kwargs_prep):
    """Pipeline listo: preprocesador del Rol D + modelo fijo del equipo."""
    return Pipeline([
        ("prep", preprocesador_d(X_train, **kwargs_prep)),
        ("modelo", modelo_base(**(modelo_kwargs or {}))),
    ])


class EliminarOutliersTrain:
    """
    Alternativa 'eliminar filas' (solo para comparar): quita de TRAIN las filas
    con algún valor fuera de los límites IQR en `columnas`. Test no se toca
    (en producción no se pueden descartar casos). Devuelve X, y filtrados.
    """

    def __init__(self, columnas, k=1.5):
        self.columnas = columnas
        self.k = k

    def aplicar(self, X, y):
        mask = pd.Series(True, index=X.index)
        for c in self.columnas:
            li, ls = limites_iqr(X[c].dropna(), self.k)
            mask &= X[c].isna() | X[c].between(li, ls)
        return X[mask], y[mask]
