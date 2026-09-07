"""
mc_common.py
============

Módulo común para los tres paquetes de entrenamiento (A, B, C) y para el
barrido de ventanas de lluvia. TODO lo que comparten los paquetes vive
aquí, de modo que una regla se corrige UNA vez y aplica a los tres:

    - carga de slope units y del inventario           (load_slope_units)
    - carga de series de pluviómetros                 (load_gauge_series)
        * duplicados de fecha-hora colapsados con max()
        * huecos como NaN + contador de cobertura (NO fill con ceros)
    - acumulados de lluvia con control de cobertura   (rain_at)
    - filtro de días húmedos: 1d > 10 mm y 30d > 50 mm
    - buffer espacial de 200 m y temporal de EXCL_DAYS
    - hold-out temporal desde HOLDOUT_START
    - asignación del pluviómetro más cercano          (nearest_gauge)
    - fusión con ONI/ENSO                             (merge_oni)
    - los tres modelos (LR, GAM, RF) con la MISMA interfaz
    - métricas, umbrales por TPR e importancia por permutación
    - el bucle Monte Carlo genérico                   (run_montecarlo)

Cada script de paquete (mc_set_a.py, mc_set_b.py, mc_set_c.py) solo
define CÓMO se construye su dataset y qué variables usa; el resto lo
hace este módulo.

DECISIONES DOCUMENTADAS
-----------------------
1. DUPLICADOS: las series traen horas repetidas por el solapamiento de
   rangos al concatenar actualizaciones. Se colapsan con
   groupby('Fecha')['P'].max(): la mayoría de duplicados son copias
   exactas y, en los pares discrepantes, uno suele ser 0 o un registro
   incompleto, de modo que max() preserva la lectura válida.
   NO usar sum(): duplica la lluvia.

2. HUECOS: un hueco de datos NO es lluvia cero. La serie se regulariza
   con asfreq('H') dejando NaN, y se lleva un contador acumulado de
   horas con dato (cv). Cada acumulado de k días exige una cobertura
   mínima (MIN_COV) dentro de su ventana; si no la cumple, devuelve NaN
   y la observación se descarta (y se cuenta cuántas se descartaron).

3. FILTRO DE DÍAS HÚMEDOS (exposición): se aplica el MISMO criterio a
   eventos y no-eventos (1d > UMBRAL_1D y 30d > UMBRAL_30D), de modo
   que ambas clases están condicionadas igual y el modelo compara
   condiciones lluviosas con y sin deslizamiento, no lluvia vs. sequía.

4. BUFFER TEMPORAL (paquete A): la exclusión es POR SLOPE UNIT
   (alrededor de los eventos de ESA unidad), no global. En el paquete C
   la exclusión sí es global (la fecha debe ser distinta de la de
   cualquier evento): run_montecarlo registra la distribución anual de
   las ausencias para vigilar el sesgo temporal que eso puede inducir.

5. RELACIÓN 2:1 y SPLIT 80/20 AGRUPADO POR SLOPE UNIT, iguales en los
   tres paquetes para que la comparación sea justa.

6. HOLD-OUT TEMPORAL: los deslizamientos ocurridos desde HOLDOUT_START
   (1 de abril de 2025) NO entran al entrenamiento ni al test, y las
   fechas candidatas de las ausencias tampoco cruzan esa fecha. La
   primera temporada de lluvias de 2025 queda así reservada como
   periodo de validación independiente para los mapas dinámicos.
   El corte se aplica en DOS sitios y los dos hacen falta:
       - load_slope_units() : filtra las presencias
       - wet_candidates()   : limita las fechas de las ausencias
   Si solo se filtraran las presencias, las ausencias se seguirían
   sorteando dentro del periodo reservado y el modelo lo vería.
   Los eventos apartados se guardan en holdout_events.csv.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

from pygam import LogisticGAM, s, f
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# CONFIGURACIÓN GLOBAL (compartida por los tres paquetes)
# ============================================================

CFG = {
    # --- rutas ---
    "PATH_SU_GPKG": "su_todas_con_inventario.gpkg",
    "LAYER":        "slope_units_cap5",   # None -> primera capa del gpkg
    "PLUV_META":    "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv",
    "RUTA_SERIES":  "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/",
    "ONI_PATH":     "/home/oisanchezp/Thesis/data/metadata/oni_diario_2010_2025.csv",
    "OUT_DIR":      "/home/oisanchezp/Thesis/src/05_Chapter/Montecarlo/",

    # --- columnas del gpkg ---
    "COL_TARGET": "mm_si",             # 1 = slope unit-fecha con deslizamiento
    "COL_FECHA":  "fecha_hora_evento",
    "STATIC_NUM": ["slope_mean", "curva_mean"],
    "STATIC_CAT": ["cobertura", "suelos"],

    # --- filtro de días húmedos (exposición), igual en SI y NO ---
    "UMBRAL_1D":  10.0,   # mm en 1 día
    "UMBRAL_30D": 50.0,   # mm en 30 días

    # --- buffers ---
    "MIN_DIST_M": 200.0,  # buffer espacial evento -> ausencia
    "EXCL_DAYS":  3,      # buffer temporal fecha evento -> fecha ausencia

    # --- hold-out temporal ---
    # Los eventos desde esta fecha se reservan para la validación
    # independiente de los mapas dinámicos (Sección 5.4.4).
    "HOLDOUT_START": "2025-04-01",

    # --- muestreo y evaluación ---
    "RATIO_NO_SI": 2,     # ausencias : presencias
    "N_MC":        100,   # corridas Monte Carlo
    "TEST_SIZE":   0.20,  # split 80/20 (mantener sincronizado con el texto)
    "BASE_SEED":   100,

    # --- lluvia ---
    "MIN_COV":  0.90,     # cobertura mínima de datos dentro de una ventana
    "H1":       24,       # horas por día

    # --- GAM ---
    "N_SPLINES": 6,
    "LAM_GRID":  np.logspace(0, 3, 5),

    # --- Random Forest (ajustar y reportar en la tesis) ---
    "RF_N_ESTIMATORS": 500,
    "RF_MIN_LEAF":     5,

    # --- umbrales operacionales ---
    "TPR_TARGETS": (0.70, 0.85, 0.95),
}

ENSO_MAP = {"La Niña": 0, "Neutro": 1, "El Niño": 2}


# ============================================================
# LOGGING
# ============================================================

def setup_logging(out_dir: str, tag: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{tag}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    logging.info("=" * 70)
    logging.info("INICIO %s: %s", tag, datetime.now().isoformat(timespec="seconds"))


# ============================================================
# 1) SLOPE UNITS E INVENTARIO
# ============================================================

# Nombres candidatos para el identificador de slope unit, por prioridad.
# El gpkg del Capítulo 4 usa 'su_id' o 'Id'; r.slopeunits suele dejar 'cat'.
SU_ID_CANDIDATOS = ("su_uid", "su_id", "SU_ID", "Id", "id", "cat", "DN", "fid")


def asignar_su_uid(su: gpd.GeoDataFrame,
                   avisar: bool = True) -> gpd.GeoDataFrame:
    """Garantiza una columna `su_uid` que identifica la UNIDAD, no la fila.

    Es una distinción crítica. El gpkg trae una fila por combinación slope
    unit-evento, así que una unidad con tres deslizamientos ocupa tres filas
    y las tres deben compartir el mismo identificador. De `su_uid` dependen:

      * el `GroupShuffleSplit`, que separa entrenamiento y prueba por unidad
        para que la misma ladera no aparezca en ambos lados;
      * el buffer temporal, que acumula por unidad las fechas excluidas.

    Si `su_uid` fuera el índice de fila, cada fila sería su propio grupo: el
    split se volvería aleatorio y el buffer quedaría incompleto. Por eso aquí
    nunca se usa el índice; si no hay identificador real se deriva de la
    geometría, que sí es estable dentro de una misma lectura del archivo.
    """
    su = su.copy()

    if "su_uid" not in su.columns:
        cand = next((c for c in SU_ID_CANDIDATOS if c in su.columns), None)
        if cand is not None:
            su["su_uid"] = su[cand]
            logging.info("su_uid tomado de la columna '%s'", cand)
        else:
            # Último recurso: huella de la geometría. Las filas duplicadas por
            # el spatial join comparten polígono, así que comparten huella.
            su["su_uid"] = pd.factorize(su.geometry.apply(lambda g: g.wkb))[0]
            logging.warning(
                "El gpkg no tiene ninguna de %s. su_uid se derivó de la "
                "geometría. Columnas disponibles: %s",
                list(SU_ID_CANDIDATOS), sorted(su.columns.tolist()))

    # Diagnóstico: si cada fila tiene su propio uid pero hay unidades con
    # varios eventos, el identificador está mal y el split no protegería nada.
    n_filas, n_uid = len(su), su["su_uid"].nunique()
    if avisar:
        logging.info("su_uid: %d filas | %d unidades distintas | "
                     "%d filas extra por unidades multi-evento",
                     n_filas, n_uid, n_filas - n_uid)
        if n_filas == n_uid and (su.get(CFG["COL_TARGET"], 0) == 1).sum() > 0:
            logging.warning(
                "su_uid es único en todas las filas. Si el join duplicó "
                "unidades con varios eventos, el identificador es incorrecto "
                "y el GroupShuffleSplit no evitaría fugas. Verifícalo.")
    return su


def load_slope_units() -> gpd.GeoDataFrame:
    """Carga el gpkg de slope units, valida columnas, CRS y hold-out."""
    su = (gpd.read_file(CFG["PATH_SU_GPKG"], layer=CFG["LAYER"])
          if CFG["LAYER"] else gpd.read_file(CFG["PATH_SU_GPKG"]))

    faltan = [c for c in [CFG["COL_TARGET"], CFG["COL_FECHA"],
                          *CFG["STATIC_NUM"], *CFG["STATIC_CAT"]]
              if c not in su.columns]
    if faltan:
        raise KeyError(
            f"Columnas ausentes en el gpkg: {faltan}. "
            f"Disponibles: {sorted(su.columns.tolist())}")

    if su.crs is None:
        raise ValueError("El gpkg no tiene CRS.")
    if su.crs.is_geographic:
        su = su.to_crs("EPSG:3116")   # metros, para los buffers

    su[CFG["COL_FECHA"]] = pd.to_datetime(su[CFG["COL_FECHA"]], errors="coerce")
    su = asignar_su_uid(su)

    # ---------------------------------------------------------------
    # HOLD-OUT TEMPORAL
    # Solo se filtran las FILAS DE EVENTO. Las filas de ausencia tienen
    # fecha NaT y son el pool espacial de los paquetes B y C, así que
    # deben conservarse todas.
    # ---------------------------------------------------------------
    corte = pd.Timestamp(CFG["HOLDOUT_START"])
    es_evento = su[CFG["COL_TARGET"]] == 1
    fuera = es_evento & (su[CFG["COL_FECHA"]] >= corte)

    if fuera.any():
        reservados = su[fuera].copy()
        os.makedirs(CFG["OUT_DIR"], exist_ok=True)
        (reservados.drop(columns="geometry", errors="ignore")
         .to_csv(os.path.join(CFG["OUT_DIR"], "holdout_events.csv"),
                 index=False))
        logging.info("Hold-out desde %s: %d eventos reservados -> %s",
                     corte.date(), int(fuera.sum()),
                     os.path.join(CFG["OUT_DIR"], "holdout_events.csv"))
        su = su[~fuera].copy()

    n_si = int((su[CFG["COL_TARGET"]] == 1).sum())
    logging.info("Slope units: %d | filas evento para entrenar: %d",
                 len(su), n_si)
    return su


def load_holdout_events() -> pd.DataFrame:
    """Lee los eventos reservados por el hold-out (validación de mapas)."""
    path = os.path.join(CFG["OUT_DIR"], "holdout_events.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe {path}. Corre primero load_slope_units().")
    df = pd.read_csv(path)
    df[CFG["COL_FECHA"]] = pd.to_datetime(df[CFG["COL_FECHA"]], errors="coerce")
    return df


# ============================================================
# 2) SERIES DE PLUVIÓMETROS
# ============================================================

def load_gauge_series(cod: int) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Serie horaria de un pluviómetro.

    Devuelve (cs, cv):
        cs : cumsum de lluvia (huecos tratados como 0 SOLO para la
             aritmética de la resta; cv recuerda dónde faltaba dato)
        cv : cumsum de horas con dato, para el control de cobertura
    """
    path = os.path.join(CFG["RUTA_SERIES"], f"H_Datos_Procesados_Est_{cod}.csv")
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path, usecols=["Fecha", "P"], parse_dates=["Fecha"])
    df = df.dropna(subset=["Fecha"])
    n_dup = int(df["Fecha"].duplicated().sum())

    # Duplicados -> max() (ver DECISIONES, punto 1). Huecos -> NaN.
    serie_h = df.groupby("Fecha")["P"].max().sort_index().asfreq("H")

    valida = serie_h.notna()
    cs = serie_h.fillna(0.0).cumsum().astype(np.float32)
    cv = valida.cumsum().astype(np.int32)

    pct = 100.0 * valida.sum() / max(len(serie_h), 1)
    logging.info("  Est. %4d: %s -> %s | %5.1f%% con dato | %d duplicados",
                 cod, serie_h.index.min().date(), serie_h.index.max().date(),
                 pct, n_dup)
    return cs, cv


def build_gauge_cache(codigos: List[int]) -> Dict[int, Tuple[pd.Series, pd.Series]]:
    """Carga y cachea (cs, cv) de todos los pluviómetros pedidos."""
    cache = {}
    for cod in sorted(set(int(c) for c in codigos if pd.notna(c))):
        cs, cv = load_gauge_series(cod)
        if cs is not None and len(cs) > 0:
            cache[cod] = (cs, cv)
    logging.info("Pluviómetros con serie: %d", len(cache))
    return cache


def _pos_at(cs: pd.Series, tstamp) -> int:
    """Posición del timestamp en el índice horario (pad hacia atrás)."""
    t = pd.Timestamp(tstamp).floor("H")
    if t in cs.index:
        return cs.index.get_loc(t)
    loc = cs.index.get_indexer([t], method="pad")
    return int(loc[0])          # -1 si t es anterior al inicio de la serie


def rain_at(cs: pd.Series, cv: pd.Series, tstamp, k_days: int) -> float:
    """Lluvia acumulada en los k_days*24 h que terminan en tstamp.

    NaN si el timestamp cae fuera de la serie, si la ventana se sale del
    registro o si la cobertura de datos es menor que MIN_COV.
    """
    if cs is None or cs.empty:
        return np.nan
    pos = _pos_at(cs, tstamp)
    lag = k_days * CFG["H1"]
    if pos < 0 or pos - lag < 0:
        return np.nan
    horas_ok = int(cv.iloc[pos] - cv.iloc[pos - lag])
    if horas_ok < CFG["MIN_COV"] * lag:
        return np.nan
    return float(cs.iloc[pos] - cs.iloc[pos - lag])


def rain_profile(cs: pd.Series, cv: pd.Series, tstamp, max_days: int = 97) -> np.ndarray:
    """Acumulados de 1..max_days días terminando en tstamp (para el barrido).

    Vectorizado: una resta de cumsum por observación. Devuelve un array
    de longitud max_days con NaN donde la cobertura no alcanza.
    """
    out = np.full(max_days, np.nan, dtype=float)
    if cs is None or cs.empty:
        return out
    pos = _pos_at(cs, tstamp)
    if pos < 0:
        return out
    lags = np.arange(1, max_days + 1) * CFG["H1"]
    ok = pos - lags >= 0
    if not ok.any():
        return out
    lags_ok = lags[ok]
    vals  = cs.iloc[pos] - cs.iloc[pos - lags_ok].to_numpy()
    horas = cv.iloc[pos] - cv.iloc[pos - lags_ok].to_numpy()
    vals = np.where(horas >= CFG["MIN_COV"] * lags_ok, vals, np.nan)
    out[ok] = vals
    return out


# ============================================================
# 3) CANDIDATOS DE FECHAS HÚMEDAS POR PLUVIÓMETRO
# ============================================================

def wet_candidates(cache: Dict[int, Tuple[pd.Series, pd.Series]],
                   excluded_dates: Optional[pd.Index] = None,
                   ) -> Dict[int, pd.DatetimeIndex]:
    """Timestamps horarios que cumplen el filtro de días húmedos.

    Para cada pluviómetro: 1d > UMBRAL_1D y 30d > UMBRAL_30D, con la
    cobertura mínima en ambas ventanas. Se descartan además las fechas
    posteriores a HOLDOUT_START (segundo punto del hold-out temporal;
    sin esto las ausencias se sortearían dentro del periodo reservado).
    Si excluded_dates no es None, se eliminan también esos días.
    """
    H1, H30 = CFG["H1"], 30 * CFG["H1"]
    corte = pd.Timestamp(CFG["HOLDOUT_START"])
    out = {}
    for cod, (cs, cv) in cache.items():
        ll_1d  = cs - cs.shift(H1)
        ll_30d = cs - cs.shift(H30)
        cov_1d  = (cv - cv.shift(H1))  >= CFG["MIN_COV"] * H1
        cov_30d = (cv - cv.shift(H30)) >= CFG["MIN_COV"] * H30

        ok = ((ll_1d > CFG["UMBRAL_1D"]) & (ll_30d > CFG["UMBRAL_30D"])
              & cov_1d & cov_30d)
        ok &= pd.Series(cs.index < corte, index=cs.index)   # hold-out temporal
        if excluded_dates is not None and len(excluded_dates) > 0:
            ok &= ~pd.Index(cs.index.normalize().date).isin(excluded_dates)

        cand = cs.index[ok.fillna(False).to_numpy()]
        if len(cand):
            out[cod] = cand
    logging.info("Pluviómetros con fechas candidatas (hasta %s): %d",
                 corte.date(), len(out))
    return out


def excluded_dates_global(fechas_evento, excl_days: int) -> pd.Index:
    """±excl_days alrededor de CADA fecha de evento (paquete C)."""
    bloques = [pd.date_range(d.normalize() - pd.Timedelta(days=excl_days),
                             d.normalize() + pd.Timedelta(days=excl_days),
                             freq="D").date
               for d in pd.to_datetime(fechas_evento).dropna()]
    return pd.Index(np.unique(np.concatenate(bloques))) if bloques else pd.Index([])


def excluded_dates_by_su(su_si: pd.DataFrame, excl_days: int) -> Dict:
    """±excl_days alrededor de los eventos de CADA slope unit (paquete A)."""
    out = {}
    for uid, g in su_si.groupby("su_uid"):
        out[uid] = excluded_dates_global(g[CFG["COL_FECHA"]], excl_days)
    return out


# ============================================================
# 4) ASIGNACIÓN DE PLUVIÓMETRO Y ONI
# ============================================================

def read_pluvios_gdf(target_crs) -> gpd.GeoDataFrame:
    pl = pd.read_csv(CFG["PLUV_META"])
    pl["FechaInstalacion"] = pd.to_datetime(pl["FechaInstalacion"], errors="coerce")
    gpl = gpd.GeoDataFrame(
        pl, geometry=gpd.points_from_xy(pl["Longitude"], pl["Latitude"]),
        crs="EPSG:4326").to_crs(target_crs)
    return gpl[["Codigo", "FechaInstalacion", "geometry"]].copy()


def nearest_gauge(su_gdf: gpd.GeoDataFrame, gpl: gpd.GeoDataFrame,
                  valid_codes: set) -> gpd.GeoDataFrame:
    """Pluviómetro más cercano al centroide de cada slope unit.

    Guarda también la distancia (dist_pluv_m): repórtala en la tesis
    (mediana y máximo) porque sostiene el supuesto de representatividad.
    """
    su_gdf = su_gdf.copy()
    gpl = gpl[gpl["Codigo"].astype(int).isin(valid_codes)].reset_index(drop=True)
    if gpl.empty:
        su_gdf["Codigo_pluvio"] = np.nan
        su_gdf["dist_pluv_m"] = np.nan
        return su_gdf

    cent = su_gdf.geometry.centroid
    joined = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=cent, crs=su_gdf.crs),
        gpl, how="left", distance_col="dist_pluv_m")
    # sjoin_nearest puede duplicar si hay empates exactos; nos quedamos con el 1o
    joined = joined[~joined.index.duplicated(keep="first")]
    su_gdf["Codigo_pluvio"] = joined["Codigo"].astype("Int64").to_numpy()
    su_gdf["dist_pluv_m"]   = joined["dist_pluv_m"].to_numpy()
    return su_gdf


def merge_oni(df: pd.DataFrame) -> pd.DataFrame:
    """Añade ONI y fase ENSO por fecha, y el código numérico ENSO_code."""
    oni = pd.read_csv(CFG["ONI_PATH"], parse_dates=["date"])
    oni["date"] = oni["date"].dt.normalize()
    df = df.copy()
    df["date"] = pd.to_datetime(df[CFG["COL_FECHA"]]).dt.normalize()
    for col in ("ONI", "ENSO"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.merge(oni[["date", "ONI", "ENSO"]], on="date", how="left").drop(columns=["date"])

    # astype("string") mantiene <NA> real: con astype(str) los faltantes
    # se convertirían en la cadena "nan" y pasarían como una categoría más.
    enso = df["ENSO"].astype("string").str.strip().replace({"Neutral": "Neutro"})
    enso = enso.where(enso.isin(list(ENSO_MAP.keys())))   # lo demás -> <NA>

    df["ENSO"] = enso
    df["ENSO_code"] = enso.map(ENSO_MAP).astype("Int64")
    n_bad = int(enso.isna().sum())
    if n_bad:
        logging.warning("  %d observaciones sin fase ENSO (fecha fuera de la "
                        "tabla ONI); se descartan en run_montecarlo", n_bad)
    return df


def add_rain(df: pd.DataFrame, cache, k_list: List[int]) -> pd.DataFrame:
    """Añade columnas '1d', '30d', ... usando el pluviómetro asignado."""
    df = df.copy()
    for k in k_list:
        df[f"{k}d"] = np.nan
    for cod, idxs in df.groupby("Codigo_pluvio").groups.items():
        if pd.isna(cod) or int(cod) not in cache:
            continue
        cs, cv = cache[int(cod)]
        for i in idxs:
            t = df.at[i, CFG["COL_FECHA"]]
            for k in k_list:
                df.at[i, f"{k}d"] = rain_at(cs, cv, t, k)
    return df


def apply_wet_filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Filtro de exposición (1d > 10, 30d > 50), registrando descartes."""
    n0 = len(df)
    df = df[(df["1d"] > CFG["UMBRAL_1D"]) & (df["30d"] > CFG["UMBRAL_30D"])].copy()
    if n0 - len(df) > 0:
        logging.info("  filtro húmedo %s: descartadas %d/%d (%.1f%%)",
                     label, n0 - len(df), n0, 100.0 * (n0 - len(df)) / max(n0, 1))
    return df


# ============================================================
# 5) MODELOS: la misma interfaz para LR, GAM y RF
# ============================================================
#
# Los tres reciben un DataFrame con las columnas de features y devuelven
# probabilidades. La regresión logística y el RF van dentro de un
# Pipeline de sklearn con one-hot para las categóricas (un código
# numérico crudo impondría un orden falso en la logística). El GAM usa
# f() para las categóricas, que es su forma nativa de tratarlas.

class GamWrapper:
    """Envuelve LogisticGAM con la interfaz fit/predict_proba(DataFrame)."""

    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.cat_maps: Dict[str, Dict] = {}
        self.gam: Optional[LogisticGAM] = None

    def _terms(self):
        n = CFG["N_SPLINES"]
        terms = s(0, n_splines=n)
        for j in range(1, len(self.num_cols)):
            terms += s(j, n_splines=n)
        for j in range(len(self.num_cols),
                       len(self.num_cols) + len(self.cat_cols)):
            terms += f(j)
        return terms

    def _to_X(self, Xdf: pd.DataFrame) -> np.ndarray:
        cols = [Xdf[c].astype(float).to_numpy() for c in self.num_cols]
        for c in self.cat_cols:
            m = self.cat_maps[c]
            cols.append(Xdf[c].astype(str).map(m).astype(float).to_numpy())
        return np.column_stack(cols)

    def fit(self, Xdf: pd.DataFrame, y: np.ndarray, seed: int = 42):
        for c in self.cat_cols:
            cats = sorted(Xdf[c].astype(str).unique())
            self.cat_maps[c] = {v: i for i, v in enumerate(cats)}
        X = self._to_X(Xdf)
        # lambda elegido SOLO dentro del train (split interno 70/30)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=seed)
        best_lam, best_auc = float(CFG["LAM_GRID"][0]), -np.inf
        for lam in CFG["LAM_GRID"]:
            g = LogisticGAM(self._terms(), lam=float(lam)).fit(X_tr, y_tr)
            auc = roc_auc_score(y_val, g.predict_proba(X_val))
            if auc > best_auc:
                best_auc, best_lam = auc, float(lam)
        self.gam = LogisticGAM(self._terms(), lam=best_lam).fit(X, y)
        self.best_lam = best_lam
        return self

    def predict_proba(self, Xdf: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.gam.predict_proba(self._to_X(Xdf)))


def make_models(num_cols: List[str], cat_cols: List[str], seed: int) -> Dict:
    """Los tres modelos con la misma interfaz DataFrame -> probas."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    rf = Pipeline([
        ("pre", ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])),
        ("clf", RandomForestClassifier(
            n_estimators=CFG["RF_N_ESTIMATORS"],
            min_samples_leaf=CFG["RF_MIN_LEAF"],
            n_jobs=-1, random_state=seed)),
    ])
    gam = GamWrapper(num_cols, cat_cols)
    return {"LR": lr, "GAM": gam, "RF": rf}


def _proba1(model, Xdf: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(Xdf)
    p = np.asarray(p)
    return p[:, 1] if p.ndim == 2 else p


# ============================================================
# 6) MÉTRICAS, UMBRALES E IMPORTANCIA
# ============================================================

def eval_metrics(y_true, y_prob, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    pofd      = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "auc":   float(roc_auc_score(y_true, y_prob)),
        "ap":    float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1":    float(f1_score(y_true, y_pred, zero_division=0)),
        "bacc":  float(balanced_accuracy_score(y_true, y_prob >= thr)),
        "recall": float(recall), "precision": float(precision),
        "pofd": float(pofd), "hk": float(recall - pofd),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def thresholds_at_tpr(y_true, y_prob) -> Dict[str, Dict[str, float]]:
    """Cortes de probabilidad en TPR objetivo + óptimo de Youden.

    OJO: el valor del corte depende de la relación 2:1 del muestreo;
    reportarlo por su TPR objetivo, no por su valor absoluto.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    out = {}
    for tt in CFG["TPR_TARGETS"]:
        idx = np.where(tpr >= tt)[0]
        i = int(idx[0]) if len(idx) else int(np.argmax(tpr))
        out[f"P_TPR{int(round(tt*100))}"] = {
            "threshold": float(thr[i]), "tpr": float(tpr[i]),
            "fpr": float(fpr[i]), "tnr": float(1 - fpr[i])}
    j = int(np.argmax(tpr - fpr))
    out["P_OPT"] = {"threshold": float(thr[j]), "tpr": float(tpr[j]),
                    "fpr": float(fpr[j]), "tnr": float(1 - fpr[j]),
                    "youden_J": float(tpr[j] - fpr[j])}
    return out


def permutation_importance(model, X_test: pd.DataFrame, y_test,
                           n_perm: int = 50, seed: int = 0) -> Dict[str, float]:
    """Caída media de AUROC al permutar cada columna del test.

    Se permuta la columna ORIGINAL del DataFrame (antes del one-hot),
    de modo que la importancia es comparable entre LR, GAM y RF.
    """
    rng = np.random.default_rng(seed)
    auc_base = roc_auc_score(y_test, _proba1(model, X_test))
    out = {}
    for col in X_test.columns:
        drops = np.empty(n_perm)
        for k in range(n_perm):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops[k] = auc_base - roc_auc_score(y_test, _proba1(model, Xp))
        out[col] = float(np.mean(drops))
    return out


# ============================================================
# 7) BUCLE MONTE CARLO GENÉRICO
# ============================================================

# Rejilla común de FPR para interpolar las curvas ROC de todas las
# corridas. Sin ella no se pueden promediar (cada corrida produce la
# curva en puntos distintos) ni dibujar la banda de incertidumbre.
FPR_GRID = np.linspace(0.0, 1.0, 101)


def run_montecarlo(tag: str, build_dataset, num_cols: List[str],
                   cat_cols: List[str], n_mc: Optional[int] = None) -> None:
    """Bucle Monte Carlo común a los tres paquetes.

    Parameters
    ----------
    tag           : nombre del paquete ('set_A', 'set_B', 'set_C')
    build_dataset : callable(seed) -> DataFrame con si_no + features.
                    CADA PAQUETE define el suyo; es la única diferencia.
    num_cols, cat_cols : columnas numéricas y categóricas del modelo.

    Salidas en CFG['OUT_DIR']:
        {tag}_mc_metrics.csv     métricas por corrida y modelo
        {tag}_mc_thresholds.csv  cortes por TPR objetivo y Youden
        {tag}_mc_importance.csv  importancia por permutación
        {tag}_mc_roc.csv         curvas ROC interpoladas (para la figura)
    """
    n_mc = n_mc or CFG["N_MC"]
    feat_cols = num_cols + cat_cols

    rows_metrics, rows_thr, rows_imp, rows_roc = [], [], [], []

    for mc in range(n_mc):
        seed = CFG["BASE_SEED"] + mc
        logging.info("=== %s | MC %d/%d | seed=%d ===", tag, mc + 1, n_mc, seed)
        try:
            df = build_dataset(seed)
            df = df.dropna(subset=["si_no"] + feat_cols).copy()
            y = df["si_no"].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                raise ValueError("falta una clase tras el filtrado")

            Xdf = df[feat_cols].copy()
            for c in cat_cols:
                Xdf[c] = Xdf[c].astype(str)

            # Split 80/20 AGRUPADO POR SLOPE UNIT: todas las filas de una
            # misma unidad caen del mismo lado, de modo que el test solo
            # contiene laderas que el modelo no vio al entrenar.
            if "su_uid" in df.columns:
                gss = GroupShuffleSplit(n_splits=1,
                                        test_size=CFG["TEST_SIZE"],
                                        random_state=seed)
                idx_tr, idx_te = next(
                    gss.split(Xdf, y, groups=df["su_uid"].to_numpy()))
                logging.info("  split agrupado | SU train=%d | SU test=%d",
                             df["su_uid"].iloc[idx_tr].nunique(),
                             df["su_uid"].iloc[idx_te].nunique())
            else:
                logging.warning("  sin columna su_uid: split aleatorio "
                                "(riesgo de fuga por unidad repetida)")
                idx_tr, idx_te = train_test_split(
                    np.arange(len(y)), test_size=CFG["TEST_SIZE"],
                    stratify=y, random_state=seed)

            # distribución anual de las ausencias (vigila el buffer temporal)
            anhos = (pd.to_datetime(df.loc[df["si_no"] == 0, CFG["COL_FECHA"]])
                     .dt.year.value_counts().sort_index().to_dict())
            logging.info("  ausencias por año: %s", anhos)

            for name, model in make_models(num_cols, cat_cols, seed).items():
                if isinstance(model, GamWrapper):
                    model.fit(Xdf.iloc[idx_tr], y[idx_tr], seed=seed)
                else:
                    model.fit(Xdf.iloc[idx_tr], y[idx_tr])

                y_prob = _proba1(model, Xdf.iloc[idx_te])
                m = eval_metrics(y[idx_te], y_prob)
                rows_metrics.append({"run_id": mc, "model": name,
                                     "n": len(y), **m})

                for lev, info in thresholds_at_tpr(y[idx_te], y_prob).items():
                    rows_thr.append({"run_id": mc, "model": name,
                                     "thr_level": lev, **info})

                # Curva ROC interpolada en la rejilla común de FPR, para
                # poder promediar las corridas y dibujar la banda.
                fpr_r, tpr_r, _ = roc_curve(y[idx_te], y_prob)
                rows_roc.append(pd.DataFrame({
                    "run_id": mc, "model": name, "fpr": FPR_GRID,
                    "tpr": np.interp(FPR_GRID, fpr_r, tpr_r)}))

                imp = permutation_importance(model, Xdf.iloc[idx_te],
                                             y[idx_te], seed=seed)
                rows_imp.append({"run_id": mc, "model": name, **imp})

                logging.info("  %-3s AUC=%.3f AP=%.3f Brier=%.3f",
                             name, m["auc"], m["ap"], m["brier"])
        except Exception as e:
            logging.exception("run %d falló: %s", mc, e)

    out = CFG["OUT_DIR"]
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows_metrics).to_csv(
        os.path.join(out, f"{tag}_mc_metrics.csv"), index=False)
    pd.DataFrame(rows_thr).to_csv(
        os.path.join(out, f"{tag}_mc_thresholds.csv"), index=False)
    pd.DataFrame(rows_imp).to_csv(
        os.path.join(out, f"{tag}_mc_importance.csv"), index=False)
    if rows_roc:
        pd.concat(rows_roc, ignore_index=True).to_csv(
            os.path.join(out, f"{tag}_mc_roc.csv"), index=False)

    met = pd.DataFrame(rows_metrics)
    if not met.empty:
        resumen = (met.groupby("model")["auc"]
                   .agg(["median", "mean", "std"]).round(4))
        logging.info("RESUMEN %s (AUROC):\n%s", tag, resumen.to_string())
    logging.info("=== %s LISTO ===", tag)
