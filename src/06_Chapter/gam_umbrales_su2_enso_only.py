"""
gam_umbrales_su2_enso_only.py
==============================

Modelo GAM para umbrales de lluvia detonadora de movimientos en masa
sin usar zonas de lluvia como variable explicativa.

ESTRUCTURA DEL MODELO
---------------------
    P(deslizamiento) = f(lluvia_1d, lluvia_30d, ENSO)

Variables:
    T            : lluvia 1 día (mm)              -> smooth s(T)
    P            : lluvia antecedente (mm)         -> smooth s(P)
                   P = lluvia_30d - lluvia_1d  (evita colinealidad)
    ENSO_code    : fase ENSO (0=La Niña, 1=Neutro, 2=El Niño) -> factor f()

PIPELINE
--------
1. Filtrar SU susceptibles con `su_type=2`.
2. Separar SI (mm_si=1) y NO (mm_si=0); buffer espacial 200 m.
3. Construir pool de pluvios y fechas candidatas no-evento que cumplan:
       lluvia_1d  >= UMBRAL_1D
       lluvia_30d >= UMBRAL_30D
       fecha fuera de ± EXCL_DAYS de un evento
4. Para cada corrida MC:
   - Muestrear NO con ratio 2:1 sobre eventos.
   - Calcular lluvia 1d y 30d para SI y NO.
   - Aplicar filtro de exposición simétrico en SI y NO.
   - Asignar ENSO desde tabla ONI.
   - Ajustar GAM, evaluar AUC/PR-AUC/Brier.
   - Extraer umbrales en TPR=0.70, 0.85, 0.95 + OPT (Youden).
   - Predecir grilla densa (lluvia_1d × lluvia_30d) para cada fase ENSO.
5. Agregar N_MC corridas: mediana e IC 90 % de métricas, umbrales y grilla.
6. Reentrenar GAM final con la corrida estrella (AUC mediana).
7. Guardar todo para graficar isolíneas.

SALIDAS en OUT_DIR
------------------
    gam_su2_mc_metrics.csv
    gam_su2_thresholds_mc.csv
    gam_su2_grid_probs.parquet
    gam_su2_grid_probs_per_run.parquet
    gam_su2_run_star.joblib
    gam_su2_package.json
    gam_su2_dataset_run_star.parquet
    gam_su2_no_events_all_runs.parquet
    gam_su2_roc_curves_mc.parquet
    gam_su2_ensayo.log

USO
---
    python gam_umbrales_su2_enso_only.py
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd

from pygam import LogisticGAM, s, f
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURACIÓN
# ============================================================

# --- Paths ---
PATH_GPKG   = "/home/oisanchezp/Thesis/data/processed/su_susceptibles_con_inventario.gpkg"
LAYER_IN    = "slope_units_eventos_parte2"  # primera capa por defecto; especificar si hay varias

PLUV_META   = "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv"
RUTA_SERIES = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"
ONI_PATH    = "/home/oisanchezp/Thesis/data/metadata/oni_diario_2010_2025.csv"

OUT_DIR     = "/home/oisanchezp/Thesis/src/06_Chapter/Umbrales/enso_only/"

# --- Filtro de tipo de SU para este ensayo ---
SU_TYPE_TARGET = 2

# --- Hiperparámetros de muestreo ---
T_DAYS    = 1     # lluvia de corto plazo
LONG_DAYS = 30    # lluvia de largo plazo

N_MC      = 50    # corridas MC
TEST_SIZE = 0.30
BASE_SEED = 100
RANDOM_STATE_SPLIT = 42

MIN_DIST_M = 200      # buffer espacial SU evento -> candidatos no-evento
EXCL_DAYS  = 30       # buffer temporal fechas evento

# Filtro de exposición para fechas candidatas no-evento
UMBRAL_1D  = 5.0      # mm en 1 día
UMBRAL_30D = 50.0     # mm en 30 días

# Ratio caso:control
RATIO_NO_SI = 2

# --- GAM ---
H1       = 24
LAM_GRID = np.logspace(0, 3, 5)
N_SPLINES_T = 8
N_SPLINES_P = 8

# Niveles de TPR para los umbrales operacionales
TPR_TARGETS = (0.70, 0.85, 0.95)

# --- Grilla para isolíneas (lluvia_1d × lluvia_30d) ---
GRID_T_MIN, GRID_T_MAX, GRID_T_N = 0.0, 150.0, 50    # mm en 1 día
GRID_30_MIN, GRID_30_MAX, GRID_30_N = 0.0, 600.0, 50  # mm en 30 días

# --- Nombres de columnas en el GPKG ---
COL_TARGET   = "mm_si"
COL_FECHA    = "fecha_hora_evento"
COL_SU_TYPE  = "su_type"
COL_SU_ID    = "fid"   # se crea si no existe

# --- Mappings categóricos fijos ---
ENSO_MAP = {"La Niña": 0, "Neutro": 1, "El Niño": 2}
ENSO_NORMALIZE = {"Neutral": "Neutro"}


# ============================================================
# LOGGING
# ============================================================

def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "gam_su2_ensayo.log")
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("=" * 70)
    logging.info("INICIO: %s", datetime.now().isoformat(timespec="seconds"))
    logging.info("OUT_DIR: %s", out_dir)


# ============================================================
# IO PLUVIÓMETROS Y SERIES
# ============================================================

def read_pluvios_as_gdf(pluv_meta_csv: str, target_crs) -> gpd.GeoDataFrame:
    pl = pd.read_csv(pluv_meta_csv)
    pl["FechaInstalacion"] = pd.to_datetime(pl["FechaInstalacion"], errors="coerce")
    gpl = gpd.GeoDataFrame(
        pl,
        geometry=gpd.points_from_xy(pl["Longitude"], pl["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(target_crs)
    return gpl[["Codigo", "FechaInstalacion", "geometry"]].copy()


def load_hourly_cumsum_for_gauge(cod: int, ruta_series: str) -> Optional[pd.Series]:
    """Carga la serie horaria del pluvio cod y devuelve el cumsum (mm)."""
    fn = f"H_Datos_Procesados_Est_{cod}.csv"
    path = os.path.join(ruta_series, fn)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["Fecha", "P"], parse_dates=["Fecha"])
    serie_h = df.groupby("Fecha")["P"].sum().sort_index().asfreq("H", fill_value=0.0)
    return serie_h.cumsum().astype(np.float32)


def compute_Rk_at_time(cs: pd.Series, tstamp, k_days: int) -> float:
    """Lluvia acumulada en las últimas k_days*24 h previas a tstamp."""
    if cs is None or cs.empty:
        return np.nan
    t = pd.Timestamp(tstamp).floor("H")
    try:
        pos = cs.index.get_loc(t)
    except KeyError:
        loc = cs.index.get_indexer([t], method="pad")
        pos = int(loc[0])
        if pos < 0:
            return np.nan
    lag = k_days * H1
    if pos - lag < 0:
        return np.nan
    return float(cs.iloc[pos] - cs.iloc[pos - lag])


def add_rain_columns_from_cache(
    df: pd.DataFrame,
    gauge_col: str,
    date_col: str,
    cache_cs: Dict[int, pd.Series],
    k_list: List[int],
) -> pd.DataFrame:
    df = df.copy()
    for k in k_list:
        df[f"{k}d"] = np.nan
    for cod, idxs in df.groupby(gauge_col).groups.items():
        if pd.isna(cod):
            continue
        cod = int(cod)
        cs = cache_cs.get(cod)
        if cs is None:
            continue
        for i in idxs:
            tstamp = df.at[i, date_col]
            for k in k_list:
                df.at[i, f"{k}d"] = compute_Rk_at_time(cs, tstamp, k)
    return df


# ============================================================
# POOL DE PLUVIOS Y FECHAS CANDIDATAS NO-EVENTO
# ============================================================

def build_excluded_dates(su_si_dates, excl_days: int = EXCL_DAYS) -> pd.Index:
    """Conjunto de fechas excluidas: ± excl_days alrededor de cada evento."""
    bloques = []
    for d in pd.to_datetime(su_si_dates).dropna():
        r = pd.date_range(
            d.normalize() - pd.Timedelta(days=excl_days),
            d.normalize() + pd.Timedelta(days=excl_days),
            freq="D",
        )
        bloques.append(r.date)
    if not bloques:
        return pd.Index([])
    return pd.Index(np.unique(np.concatenate(bloques)))


def build_gauge_pool(
    pluv_meta_csv: str,
    ruta_series: str,
    excluded_dates: pd.Index,
    umbral_1d: float = UMBRAL_1D,
    umbral_30d: float = UMBRAL_30D,
    min_days_after_install: int = LONG_DAYS,
) -> Tuple[Dict[int, pd.Series], Dict[int, pd.DatetimeIndex]]:
    """
    Cachea las series cumsum y devuelve por pluvio el conjunto de fechas
    horarias que cumplen filtros de exposición y exclusión temporal.
    """
    pl = pd.read_csv(pluv_meta_csv)
    pl["FechaInstalacion"] = pd.to_datetime(pl["FechaInstalacion"], errors="coerce")
    cache_cs: Dict[int, pd.Series] = {}
    candidates: Dict[int, pd.DatetimeIndex] = {}

    H_T  = T_DAYS    * 24
    H_30 = LONG_DAYS * 24

    for _, row in pl.iterrows():
        cod = int(row["Codigo"])
        inst = row["FechaInstalacion"]
        cs = load_hourly_cumsum_for_gauge(cod, ruta_series)
        if cs is None or cs.empty:
            continue
        cache_cs[cod] = cs
        idx = cs.index

        ll_1d  = cs - cs.shift(H_T)
        ll_30d = cs - cs.shift(H_30)

        ok = (ll_1d >= umbral_1d) & (ll_30d >= umbral_30d)
        ok = ok & (~pd.Index(idx.normalize().date).isin(excluded_dates))
        if pd.notna(inst):
            ok = ok & (idx >= (inst + pd.Timedelta(days=min_days_after_install)))

        cand = idx[ok.fillna(False)]
        if len(cand) > 0:
            candidates[cod] = cand

    return cache_cs, candidates


def assign_nearest_gauge_with_candidates(
    su_gdf: gpd.GeoDataFrame,
    gpl: gpd.GeoDataFrame,
    candidates_by_gauge: Dict[int, pd.DatetimeIndex],
) -> gpd.GeoDataFrame:
    """Asigna a cada SU el pluvio más cercano que tenga fechas candidatas."""
    su_gdf = su_gdf.copy()
    gpl = gpl[gpl["Codigo"].astype(int).isin(set(candidates_by_gauge.keys()))].copy()
    if gpl.empty:
        su_gdf["Codigo_pluvio"] = np.nan
        su_gdf["dist_pluv_m"]   = np.nan
        return su_gdf

    chosen_codes, chosen_dist = [], []
    gpl_geom = gpl.geometry
    for geom in su_gdf.geometry:
        if geom is None or geom.is_empty:
            chosen_codes.append(np.nan)
            chosen_dist.append(np.nan)
            continue
        ref_geom = geom if geom.geom_type == "Point" else geom.centroid
        dists = gpl_geom.distance(ref_geom)
        j = int(dists.idxmin())
        chosen_codes.append(int(gpl.loc[j, "Codigo"]))
        chosen_dist.append(float(dists.loc[j]))

    su_gdf["Codigo_pluvio"] = chosen_codes
    su_gdf["dist_pluv_m"]   = chosen_dist
    return su_gdf


# ============================================================
# CONSTRUCCIÓN DEL DATASET POR CORRIDA MC
# ============================================================

def construir_dataset_mc(
    seed: int,
    su_susceptibles: gpd.GeoDataFrame,
    cache_cs: Dict[int, pd.Series],
    candidates_by_gauge: Dict[int, pd.DatetimeIndex],
    gpl: gpd.GeoDataFrame,
    oni: pd.DataFrame,
    ratio_no_si: int = RATIO_NO_SI,
    t_days: int = T_DAYS,
    long_days: int = LONG_DAYS,
    umbral_1d: float = UMBRAL_1D,
    umbral_30d: float = UMBRAL_30D,
) -> pd.DataFrame:
    """
    Construye el dataset (SI + NO) para una corrida MC.

    Returns
    -------
    DataFrame con columnas principales:
        si_no, fecha_hora_evento, Codigo_pluvio, dist_pluv_m,
        '1d', '30d', ENSO, ENSO_code, su_type, fid
    """
    rng = np.random.default_rng(seed)

    su = su_susceptibles[su_susceptibles[COL_SU_TYPE] == SU_TYPE_TARGET].copy()
    su_si = su[su[COL_TARGET] == 1].copy()
    su_no = su[su[COL_TARGET] == 0].copy()

    if len(su_si) == 0:
        raise ValueError(f"No hay eventos en su_type={SU_TYPE_TARGET}.")
    if len(su_no) == 0:
        raise ValueError(f"No hay no-eventos en su_type={SU_TYPE_TARGET}.")

    n_si = len(su_si)
    n_no_target = ratio_no_si * n_si

    buffer_union = su_si.geometry.buffer(MIN_DIST_M).unary_union
    su_no_eligible = su_no[~su_no.geometry.intersects(buffer_union)].copy()

    if len(su_no_eligible) < n_no_target:
        raise ValueError(
            f"NO elegibles insuficientes: requieres {n_no_target}, hay {len(su_no_eligible)}."
        )

    oversample = min(len(su_no_eligible), int(n_no_target * 4))
    sampled_idx = rng.choice(su_no_eligible.index.values, size=oversample, replace=False)
    df_no = su_no_eligible.loc[sampled_idx].copy().reset_index(drop=True)
    df_no = assign_nearest_gauge_with_candidates(df_no, gpl, candidates_by_gauge)
    df_no = df_no.dropna(subset=["Codigo_pluvio"]).copy()

    if len(df_no) < n_no_target:
        raise ValueError("NO-eventos insuficientes después de asignar pluvio.")
    df_no = df_no.iloc[:n_no_target].copy()
    df_no["Codigo_pluvio"] = df_no["Codigo_pluvio"].astype(int)

    fechas = []
    for cod in df_no["Codigo_pluvio"].to_numpy():
        cand = candidates_by_gauge.get(int(cod))
        fechas.append(pd.Timestamp(rng.choice(cand.values)))
    df_no[COL_FECHA] = pd.to_datetime(fechas, errors="coerce")
    df_no = df_no.dropna(subset=[COL_FECHA]).copy()

    gpl_with_series = gpl[gpl["Codigo"].astype(int).isin(set(cache_cs.keys()))].copy()
    if gpl_with_series.empty:
        raise ValueError("No hay pluvios con series disponibles.")

    su_si = su_si.reset_index(drop=True)
    su_si[COL_FECHA] = pd.to_datetime(su_si[COL_FECHA], errors="coerce")

    codes_out, dist_out = [], []
    gpl_geom = gpl_with_series.geometry
    for geom, t_event in zip(su_si.geometry, su_si[COL_FECHA]):
        if geom is None or geom.is_empty or pd.isna(t_event):
            codes_out.append(np.nan)
            dist_out.append(np.nan)
            continue
        ref_geom = geom if geom.geom_type == "Point" else geom.centroid
        dists = gpl_geom.distance(ref_geom)
        idx_order = dists.nsmallest(15).index
        picked, picked_dist = None, None
        for j in idx_order:
            inst = gpl_with_series.loc[j, "FechaInstalacion"]
            if pd.isna(inst) or (t_event >= (inst + pd.Timedelta(days=long_days))):
                picked      = int(gpl_with_series.loc[j, "Codigo"])
                picked_dist = float(dists.loc[j])
                break
        codes_out.append(picked if picked is not None else np.nan)
        dist_out.append(picked_dist if picked_dist is not None else np.nan)

    su_si["Codigo_pluvio"] = codes_out
    su_si["dist_pluv_m"]   = dist_out
    su_si = su_si.dropna(subset=["Codigo_pluvio"]).copy()
    su_si["Codigo_pluvio"] = su_si["Codigo_pluvio"].astype(int)

    k_list = [t_days, long_days]
    su_si = add_rain_columns_from_cache(su_si, "Codigo_pluvio", COL_FECHA, cache_cs, k_list)
    df_no = add_rain_columns_from_cache(df_no, "Codigo_pluvio", COL_FECHA, cache_cs, k_list)

    n_si_pre = len(su_si)
    su_si = su_si[
        (su_si[f"{t_days}d"]    >= umbral_1d) &
        (su_si[f"{long_days}d"] >= umbral_30d)
    ].copy()
    n_si_dropped = n_si_pre - len(su_si)
    if n_si_dropped > 0:
        logging.info(
            "[seed=%d] Eventos descartados por filtro exposición "
            "(1d>=%.1f, 30d>=%.1f): %d / %d (%.1f%%)",
            seed, umbral_1d, umbral_30d,
            n_si_dropped, n_si_pre,
            100.0 * n_si_dropped / max(n_si_pre, 1),
        )

    df_no = df_no[
        (df_no[f"{t_days}d"]    >= umbral_1d) &
        (df_no[f"{long_days}d"] >= umbral_30d)
    ].copy()

    oni = oni.copy()
    oni["date"] = pd.to_datetime(oni["date"]).dt.normalize()
    oni_min, oni_max = oni["date"].min(), oni["date"].max()

    su_si = su_si[su_si[COL_FECHA].dt.normalize().between(oni_min, oni_max)].copy()
    df_no = df_no[df_no[COL_FECHA].dt.normalize().between(oni_min, oni_max)].copy()

    for dff in (su_si, df_no):
        for col in ("ONI", "ENSO"):
            if col in dff.columns:
                dff.drop(columns=[col], inplace=True)
        dff["date"] = dff[COL_FECHA].dt.normalize()
    su_si = su_si.merge(oni[["date", "ONI", "ENSO"]], on="date", how="left").drop(columns=["date"])
    df_no = df_no.merge(oni[["date", "ONI", "ENSO"]], on="date", how="left").drop(columns=["date"])

    su_si["si_no"] = 1
    df_no["si_no"] = 0

    df_mc = pd.concat([su_si, df_no], ignore_index=True)

    df_mc["ENSO"] = df_mc["ENSO"].astype(str).str.strip().replace(ENSO_NORMALIZE)
    df_mc["ENSO_code"] = df_mc["ENSO"].map(ENSO_MAP).astype("Int64")

    needed = ["si_no", COL_FECHA, "Codigo_pluvio", f"{t_days}d", f"{long_days}d", "ENSO_code"]
    df_mc = df_mc.dropna(subset=needed).copy()
    df_mc["si_no"] = df_mc["si_no"].astype(int)
    df_mc["ENSO_code"] = df_mc["ENSO_code"].astype(int)

    # Evita que `zona` quede en las salidas y se reutilice accidentalmente.
    if "zona" in df_mc.columns:
        df_mc = df_mc.drop(columns=["zona"])
    if "zona_code" in df_mc.columns:
        df_mc = df_mc.drop(columns=["zona_code"])

    logging.info(
        "[seed=%d] su_type=%d | SI=%d | NO_eligible=%d | NO_final=%d | total=%d",
        seed, SU_TYPE_TARGET, len(su_si), len(su_no_eligible), len(df_no), len(df_mc),
    )
    return df_mc


# ============================================================
# MAPPINGS Y TRANSFORM
# ============================================================

def fit_category_mappings(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Mappings string→int para variables categóricas usadas por el modelo."""
    return {"ENSO": dict(ENSO_MAP)}


def transform_df_to_X(
    df: pd.DataFrame,
    mappings: Dict[str, Dict[str, int]],
    t_days: int = T_DAYS,
    long_days: int = LONG_DAYS,
    strict: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Construye X = [T, P, ENSO_code] para el GAM.

    T = lluvia_1d
    P = lluvia_30d - lluvia_1d   (lluvia antecedente sin solapamiento)
    """
    df = df.copy()
    col_T  = f"{t_days}d"
    col_30 = f"{long_days}d"
    df["T"] = df[col_T].astype(float)
    df["P"] = df[col_30].astype(float) - df[col_T].astype(float)
    df["P"] = df["P"].clip(lower=0.0)

    df["ENSO"] = df["ENSO"].astype(str).str.strip().replace(ENSO_NORMALIZE)
    df["ENSO_code"] = df["ENSO"].map(mappings["ENSO"])

    if strict:
        faltan = {col: int(df[col].isna().sum())
                  for col in ["ENSO_code"]
                  if int(df[col].isna().sum()) > 0}
        if faltan:
            raise ValueError(f"Categorías no vistas: {faltan}")

    X = np.column_stack([
        df["T"].to_numpy(dtype=float),
        df["P"].to_numpy(dtype=float),
        df["ENSO_code"].to_numpy(dtype=float),
    ])
    return X, df


# Estructura de TERMS: índices 0=T, 1=P, 2=ENSO_code
TERMS = (
    s(0, n_splines=N_SPLINES_T) +
    s(1, n_splines=N_SPLINES_P) +
    f(2)
)
FEATURE_NAMES = ["T", "P", "ENSO_code"]


# ============================================================
# AJUSTE DEL GAM Y MÉTRICAS
# ============================================================

def get_best_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE_SPLIT,
) -> float:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=random_state
    )
    best_lam = float(LAM_GRID[0])
    best_auc = -np.inf
    for lam in LAM_GRID:
        gam = LogisticGAM(TERMS, lam=float(lam)).fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, gam.predict_proba(X_val))
        if auc > best_auc:
            best_auc = auc
            best_lam = float(lam)
    return best_lam


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pofd      = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "auc":       float(roc_auc_score(y_true, y_prob)),
        "ap":        float(average_precision_score(y_true, y_prob)),
        "brier":     float(brier_score_loss(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "bacc":      float(balanced_accuracy_score(y_true, y_pred)),
        "recall":    float(recall),
        "precision": float(precision),
        "pofd":      float(pofd),
        "hk":        float(recall - pofd),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def thresholds_at_tpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tpr_targets: Tuple[float, ...] = TPR_TARGETS,
) -> Dict[str, Dict[str, float]]:
    """Para cada TPR objetivo, devuelve umbral P*, FPR y TNR alcanzados."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    out: Dict[str, Dict[str, float]] = {}
    for tt in tpr_targets:
        idx = np.where(tpr >= tt)[0]
        i = int(idx[0]) if len(idx) > 0 else int(np.argmax(tpr))
        out[f"P_TPR{int(round(tt * 100))}"] = {
            "threshold": float(thr[i]),
            "tpr":       float(tpr[i]),
            "fpr":       float(fpr[i]),
            "tnr":       float(1.0 - fpr[i]),
        }

    j = int(np.argmax(tpr - fpr))
    out["P_OPT"] = {
        "threshold": float(thr[j]),
        "tpr":       float(tpr[j]),
        "fpr":       float(fpr[j]),
        "tnr":       float(1.0 - fpr[j]),
        "youden_J":  float(tpr[j] - fpr[j]),
    }
    return out


# ============================================================
# UNA CORRIDA MC
# ============================================================

def train_one_run(df_mc: pd.DataFrame, run_id: int) -> Dict:
    mappings = fit_category_mappings(df_mc)
    X_all, df_tr = transform_df_to_X(df_mc, mappings, t_days=T_DAYS, long_days=LONG_DAYS)
    y_all = df_tr["si_no"].to_numpy(dtype=int)

    idx = np.arange(len(y_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE_SPLIT
    )
    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_train, y_test = y_all[idx_train], y_all[idx_test]

    best_lam = get_best_lambda(X_train, y_train)
    gam = LogisticGAM(TERMS, lam=best_lam).fit(X_train, y_train)

    y_prob_test = gam.predict_proba(X_test)
    metrics = eval_metrics(y_test, y_prob_test, thr=0.5)
    thr_dict = thresholds_at_tpr(y_test, y_prob_test, tpr_targets=TPR_TARGETS)

    return {
        "run_id":   int(run_id),
        "gam":      gam,
        "mappings": mappings,
        "df_mc":    df_mc,
        "X_all":    X_all,
        "y_all":    y_all,
        "best_lam": float(best_lam),
        "metrics":  metrics,
        "thresholds": thr_dict,
        "n_obs":    int(len(y_all)),
        "y_test":      y_test.astype(int),
        "y_prob_test": y_prob_test.astype(float),
    }


# ============================================================
# PREDICCIÓN EN GRILLA (PARA ISOLÍNEAS)
# ============================================================

def predict_grid_for_run(
    gam: LogisticGAM,
    mappings: Dict[str, Dict[str, int]],
    run_id: int,
    grid_t: np.ndarray,
    grid_30: np.ndarray,
) -> pd.DataFrame:
    """
    Predice probabilidad sobre una grilla densa (lluvia_1d × lluvia_30d)
    para cada fase ENSO.

    Returns
    -------
    DataFrame largo con: run_id, ENSO, lluvia_1d, lluvia_30d, prob.
    Sólo se conservan puntos con lluvia_30d >= lluvia_1d.
    """
    rows: List[Dict] = []

    enso_inv = {v: k for k, v in mappings["ENSO"].items()}

    GG_T, GG_30 = np.meshgrid(grid_t, grid_30, indexing="xy")
    mask_valid = GG_30 >= GG_T
    T_flat  = GG_T[mask_valid]
    L30_flat = GG_30[mask_valid]
    P_flat  = (L30_flat - T_flat).clip(min=0.0)

    for ecode in sorted(mappings["ENSO"].values()):
        X_grid = np.column_stack([
            T_flat,
            P_flat,
            np.full_like(T_flat, ecode, dtype=float),
        ])
        prob = gam.predict_proba(X_grid)
        for tt, ll30, p in zip(T_flat, L30_flat, prob):
            rows.append({
                "run_id":     int(run_id),
                "ENSO":       enso_inv[ecode],
                "lluvia_1d":  float(tt),
                "lluvia_30d": float(ll30),
                "prob":       float(p),
            })

    return pd.DataFrame(rows)


# ============================================================
# AGREGACIÓN DE CORRIDAS
# ============================================================

def summarize_runs(run_outputs: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Devuelve:
        results_df : una fila por corrida (métricas + best_lam + n_obs)
        thr_df     : una fila por (corrida × umbral) con P*, TPR, FPR
        summary    : dict con métricas agregadas y run_star (AUC mediana)
    """
    rows_metrics = []
    rows_thr = []
    for out in run_outputs:
        rows_metrics.append({
            "run_id":   out["run_id"],
            "n_obs":    out["n_obs"],
            "best_lam": out["best_lam"],
            **out["metrics"],
        })
        for thr_name, thr_info in out["thresholds"].items():
            rows_thr.append({
                "run_id":     out["run_id"],
                "thr_level":  thr_name,
                "threshold": thr_info["threshold"],
                "tpr":        thr_info["tpr"],
                "fpr":        thr_info["fpr"],
                "tnr":        thr_info["tnr"],
            })

    results_df = pd.DataFrame(rows_metrics).sort_values("run_id").reset_index(drop=True)
    thr_df     = pd.DataFrame(rows_thr).sort_values(["run_id", "thr_level"]).reset_index(drop=True)

    auc_median  = float(results_df["auc"].median())
    idx_star    = (results_df["auc"] - auc_median).abs().idxmin()
    run_id_star = int(results_df.loc[idx_star, "run_id"])

    thr_summary: Dict[str, Dict[str, float]] = {}
    for thr_name in thr_df["thr_level"].unique():
        sub = thr_df[thr_df["thr_level"] == thr_name]
        thr_summary[thr_name] = {
            "median": float(sub["threshold"].median()),
            "p05":    float(sub["threshold"].quantile(0.05)),
            "p95":    float(sub["threshold"].quantile(0.95)),
            "mean":   float(sub["threshold"].mean()),
            "std":    float(sub["threshold"].std()),
            "tpr_median": float(sub["tpr"].median()),
            "fpr_median": float(sub["fpr"].median()),
        }

    summary = {
        "auc_median":  auc_median,
        "auc_mean":    float(results_df["auc"].mean()),
        "auc_std":     float(results_df["auc"].std()),
        "ap_median":   float(results_df["ap"].median()),
        "brier_median":float(results_df["brier"].median()),
        "run_id_star": run_id_star,
        "thresholds":  thr_summary,
    }
    return results_df, thr_df, summary


def aggregate_grid(grid_runs: pd.DataFrame) -> pd.DataFrame:
    """Agrega la grilla MC: mediana, p05, p95 por (ENSO, T, 30d)."""
    g = (
        grid_runs
        .groupby(["ENSO", "lluvia_1d", "lluvia_30d"], as_index=False)
        .agg(
            prob_median=("prob", "median"),
            prob_p05   =("prob", lambda x: x.quantile(0.05)),
            prob_p95   =("prob", lambda x: x.quantile(0.95)),
            prob_mean  =("prob", "mean"),
            prob_std   =("prob", "std"),
        )
    )
    return g


# ============================================================
# REENTRENAMIENTO FINAL CON LA CORRIDA ESTRELLA
# ============================================================

def retrain_run_star(run_star: Dict, summary: Dict) -> Tuple[LogisticGAM, Dict]:
    df_mc    = run_star["df_mc"].copy()
    mappings = run_star["mappings"]
    best_lam = run_star["best_lam"]
    X_all, _ = transform_df_to_X(df_mc, mappings, t_days=T_DAYS, long_days=LONG_DAYS)
    y_all = df_mc["si_no"].astype(int).to_numpy()

    gam_final = LogisticGAM(TERMS, lam=best_lam).fit(X_all, y_all)

    package = {
        "model_type":     "GAM_su2_enso_only",
        "feature_names":  FEATURE_NAMES,
        "t_days":         int(T_DAYS),
        "long_days":      int(LONG_DAYS),
        "best_lam":       float(best_lam),
        "su_type_target": int(SU_TYPE_TARGET),
        "tpr_targets":    list(TPR_TARGETS),
        "thresholds":     summary["thresholds"],
        "mappings":       mappings,
        "config": {
            "min_dist_m":      float(MIN_DIST_M),
            "excl_days":       int(EXCL_DAYS),
            "umbral_1d":       float(UMBRAL_1D),
            "umbral_30d":      float(UMBRAL_30D),
            "ratio_no_si":     int(RATIO_NO_SI),
            "test_size":       float(TEST_SIZE),
            "random_state":    int(RANDOM_STATE_SPLIT),
            "base_seed":       int(BASE_SEED),
            "n_mc":            int(N_MC),
            "lam_grid":        LAM_GRID.tolist(),
            "n_splines_T":     int(N_SPLINES_T),
            "n_splines_P":     int(N_SPLINES_P),
        },
        "notes": {
            "P_definition":  "P = lluvia_30d - lluvia_1d (sin solapamiento)",
            "ENSO_handling": "factor fijo categórico",
            "rainfall_zone_handling": "zona de lluvia excluida del modelo y de las predicciones",
        },
    }
    return gam_final, package


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging(OUT_DIR)
    logging.info("Modelo: GAM Parte 2 — su_type=%d | variables: lluvia + ENSO", SU_TYPE_TARGET)
    logging.info("N_MC=%d | RATIO_NO_SI=%d | EXCL_DAYS=%d | MIN_DIST_M=%d",
                 N_MC, RATIO_NO_SI, EXCL_DAYS, MIN_DIST_M)
    logging.info("Filtro exposición: 1d>=%.1f mm | 30d>=%.1f mm",
                 UMBRAL_1D, UMBRAL_30D)

    # --- 1. Carga estática ---
    logging.info("Cargando GPKG: %s", PATH_GPKG)
    su = gpd.read_file(PATH_GPKG, layer=LAYER_IN) if LAYER_IN else gpd.read_file(PATH_GPKG)
    if su.crs is None:
        raise ValueError("La capa no tiene CRS definido.")
    if su.crs.is_geographic:
        logging.warning("CRS geográfico detectado; reproyectando a EPSG:32618 (UTM 18N).")
        su = su.to_crs("EPSG:32618")
    target_crs = su.crs

    if COL_SU_ID not in su.columns:
        su = su.reset_index().rename(columns={"index": COL_SU_ID})
    su[COL_FECHA] = pd.to_datetime(su[COL_FECHA], errors="coerce")

    for c in [COL_TARGET, COL_SU_TYPE]:
        if c not in su.columns:
            raise KeyError(f"Falta columna '{c}' en el GPKG.")

    logging.info("SU total: %d | tipos: %s",
                 len(su), sorted(su[COL_SU_TYPE].dropna().unique().tolist()))

    su_target = su[su[COL_SU_TYPE] == SU_TYPE_TARGET].copy()
    su_si_target = su_target[su_target[COL_TARGET] == 1].copy()
    logging.info("su_type=%d | total=%d | eventos=%d | no-eventos=%d",
                 SU_TYPE_TARGET, len(su_target),
                 len(su_si_target), len(su_target) - len(su_si_target))

    excluidas = build_excluded_dates(su_si_target[COL_FECHA], excl_days=EXCL_DAYS)
    logging.info("Fechas excluidas: %d", len(excluidas))

    # --- 2. Pool de pluvios y series cacheadas ---
    logging.info("Construyendo pool de pluvios y fechas candidatas...")
    cache_cs, candidates_by_gauge = build_gauge_pool(
        PLUV_META, RUTA_SERIES, excluidas,
        umbral_1d=UMBRAL_1D, umbral_30d=UMBRAL_30D,
        min_days_after_install=LONG_DAYS,
    )
    logging.info("Pluvios con series: %d | con fechas candidatas: %d",
                 len(cache_cs), len(candidates_by_gauge))

    gpl = read_pluvios_as_gdf(PLUV_META, target_crs=target_crs)
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])

    # --- 3. Bucle Monte Carlo ---
    grid_t  = np.linspace(GRID_T_MIN,  GRID_T_MAX,  GRID_T_N)
    grid_30 = np.linspace(GRID_30_MIN, GRID_30_MAX, GRID_30_N)

    run_outputs: List[Dict] = []
    grid_dfs:    List[pd.DataFrame] = []
    no_event_dfs: List[pd.DataFrame] = []
    roc_dfs:     List[pd.DataFrame] = []

    for mc in range(N_MC):
        seed = BASE_SEED + mc
        logging.info("=== MC %d/%d | seed=%d ===", mc + 1, N_MC, seed)
        try:
            df_mc = construir_dataset_mc(
                seed=seed,
                su_susceptibles=su,
                cache_cs=cache_cs,
                candidates_by_gauge=candidates_by_gauge,
                gpl=gpl,
                oni=oni,
            )
            out = train_one_run(df_mc, run_id=mc)
            run_outputs.append(out)
            logging.info(
                "AUC=%.3f | AP=%.3f | Brier=%.3f | "
                "P*70=%.3f | P*OPT=%.3f (TPR=%.2f) | P*85=%.3f | P*95=%.3f",
                out["metrics"]["auc"], out["metrics"]["ap"], out["metrics"]["brier"],
                out["thresholds"]["P_TPR70"]["threshold"],
                out["thresholds"]["P_OPT"]["threshold"],
                out["thresholds"]["P_OPT"]["tpr"],
                out["thresholds"]["P_TPR85"]["threshold"],
                out["thresholds"]["P_TPR95"]["threshold"],
            )

            grid_df = predict_grid_for_run(
                out["gam"], out["mappings"], mc, grid_t, grid_30
            )
            grid_dfs.append(grid_df)

            roc_df = pd.DataFrame({
                "run_id": int(mc),
                "y_true": out["y_test"],
                "y_prob": out["y_prob_test"],
            })
            roc_dfs.append(roc_df)

            cols_keep = ["si_no", f"{T_DAYS}d", f"{LONG_DAYS}d", "ENSO"]
            df_no_run = (
                df_mc[df_mc["si_no"] == 0][cols_keep]
                .copy()
                .rename(columns={f"{T_DAYS}d": "1d", f"{LONG_DAYS}d": "30d"})
            )
            df_no_run["run_id"] = int(mc)
            no_event_dfs.append(df_no_run)

        except Exception as e:
            logging.exception("Run %d falló: %s", mc, e)

    if len(run_outputs) == 0:
        raise RuntimeError("No hubo corridas Monte Carlo exitosas.")

    # --- 4. Agregación y guardado ---
    results_df, thr_df, summary = summarize_runs(run_outputs)
    grid_runs = pd.concat(grid_dfs, ignore_index=True)
    grid_agg  = aggregate_grid(grid_runs)

    results_csv = os.path.join(OUT_DIR, "gam_su2_mc_metrics.csv")
    thr_csv     = os.path.join(OUT_DIR, "gam_su2_thresholds_mc.csv")
    grid_pq     = os.path.join(OUT_DIR, "gam_su2_grid_probs.parquet")
    grid_runs_pq= os.path.join(OUT_DIR, "gam_su2_grid_probs_per_run.parquet")
    star_pq     = os.path.join(OUT_DIR, "gam_su2_dataset_run_star.parquet")
    no_all_pq   = os.path.join(OUT_DIR, "gam_su2_no_events_all_runs.parquet")
    roc_pq      = os.path.join(OUT_DIR, "gam_su2_roc_curves_mc.parquet")
    model_path  = os.path.join(OUT_DIR, "gam_su2_run_star.joblib")
    pkg_path    = os.path.join(OUT_DIR, "gam_su2_package.json")

    results_df.to_csv(results_csv, index=False)
    thr_df.to_csv(thr_csv, index=False)
    grid_agg.to_parquet(grid_pq, index=False)
    grid_runs.to_parquet(grid_runs_pq, index=False)

    if no_event_dfs:
        no_all = pd.concat(no_event_dfs, ignore_index=True)
        no_all.to_parquet(no_all_pq, index=False)
        logging.info("NO acumulados de %d corridas: %d filas guardadas en %s",
                     len(no_event_dfs), len(no_all), no_all_pq)

    if roc_dfs:
        roc_all = pd.concat(roc_dfs, ignore_index=True)
        roc_all.to_parquet(roc_pq, index=False)
        logging.info("Curvas ROC de %d corridas: %d filas guardadas en %s",
                     len(roc_dfs), len(roc_all), roc_pq)

    logging.info("AUC mediana=%.4f | AP mediana=%.4f | Brier mediana=%.4f",
                 summary["auc_median"], summary["ap_median"], summary["brier_median"])
    logging.info("run_star=%d", summary["run_id_star"])

    run_star = next(r for r in run_outputs if r["run_id"] == summary["run_id_star"])
    gam_final, package = retrain_run_star(run_star, summary)
    joblib.dump(gam_final, model_path)

    package["summary_mc"]      = summary
    package["successful_runs"] = int(len(run_outputs))

    with open(pkg_path, "w", encoding="utf-8") as fjson:
        json.dump(package, fjson, ensure_ascii=False, indent=2, default=float)

    df_star = run_star["df_mc"].copy()
    for col in ["geometry", "zona", "zona_code"]:
        if col in df_star.columns:
            df_star = df_star.drop(columns=[col])
    df_star.to_parquet(star_pq, index=False)

    logging.info("=== LISTO ===")
    logging.info("Métricas:     %s", results_csv)
    logging.info("Thresholds:   %s", thr_csv)
    logging.info("Grilla agg:   %s", grid_pq)
    logging.info("Grilla x run: %s", grid_runs_pq)
    logging.info("Modelo final: %s", model_path)
    logging.info("Package:      %s", pkg_path)
    logging.info("Dataset star: %s", star_pq)


if __name__ == "__main__":
    main()
