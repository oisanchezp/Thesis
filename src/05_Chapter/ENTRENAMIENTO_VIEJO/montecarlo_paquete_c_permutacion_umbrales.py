import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from pygam import LogisticGAM, s, f
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score,
    balanced_accuracy_score, average_precision_score
)

from sklearn.metrics import roc_curve

# ============================================================
# PAQUETE C
# - NO: lugar distinto (>min_dist_m) y fecha distinta (fuera ±excl_days)
# - NO: umbrales mínimos 1d,7d,90d
# - Variables: T,P,sin_doy,cos_doy,ENSO,geologia,cobertura,pendiente,log_area
# - Modelos: GAM + Logit + RF
# - Se guarda: resultados MC, importancias por permutación (GAM/Logit),
#              y JSON con corrida cercana a mediana (para efectos parciales).
# ============================================================

# ============================================================
# 0) PARÁMETROS
# ============================================================
path_gpkg = "/home/oisanchezp/Thesis/data/metadata/slope_units_con_inventario.gpkg"
layer_in  = "slope_units_full"

pluv_meta   = "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv"
ruta_series = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"
oni_path    = "/home/oisanchezp/Thesis/data/metadata/oni_diario_2010_2025.csv"

MIN_DIST_M = 300
EXCL_DAYS  = 30

UMBRAL_1D  = 10
UMBRAL_7D  = 20
UMBRAL_90D = 80

T_DAYS     = 1
P_DAYS     = 33
TOTAL_DAYS = T_DAYS + P_DAYS  # 34

H1 = 24

# ============================================================
# 1) UTILIDADES: lluvia / pluvios
# ============================================================

def read_pluvios_as_gdf(pluv_meta_csv, target_crs):
    pl = pd.read_csv(pluv_meta_csv)
    pl["FechaInstalacion"] = pd.to_datetime(pl["FechaInstalacion"], errors="coerce")

    gpl = gpd.GeoDataFrame(
        pl,
        geometry=gpd.points_from_xy(pl["Longitude"], pl["Latitude"]),
        crs="EPSG:4326"
    ).to_crs(target_crs)

    return gpl[["Codigo", "FechaInstalacion", "geometry"]].copy()


def load_hourly_cumsum_for_gauge(cod, ruta_series):
    fn = f"H_Datos_Procesados_Est_{cod}.csv"
    path = os.path.join(ruta_series, fn)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, usecols=["Fecha", "P"], parse_dates=["Fecha"])
    serie_h = (df.groupby("Fecha")["P"].sum()
                 .sort_index()
                 .asfreq("H", fill_value=0.0))

    return serie_h.cumsum().astype(np.float32)


def compute_Rk_at_time(cs, tstamp, k_days):
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


def add_rain_columns_from_cache(df, gauge_col, date_col, cache_cs, k_list):
    """
    Igual que add_rain_columns del paquete B, pero usando cache_cs ya precargado.
    """
    df = df.copy()
    for k in k_list:
        df[f"{k}d"] = np.nan

    # ojo: si hay NaNs en gauge_col, groupby los ignora
    for cod, idxs in df.groupby(gauge_col).groups.items():
        cod = int(cod) if pd.notna(cod) else cod
        if pd.isna(cod):
            continue

        cs = cache_cs.get(cod)
        if cs is None:
            continue

        for i in idxs:
            tstamp = df.at[i, date_col]
            for k in k_list:
                df.at[i, f"{k}d"] = compute_Rk_at_time(cs, tstamp, k)

    return df


# ============================================================
# 2) CANDIDATOS (timestamps) por pluvio con umbrales + exclusión ±excl_days
# ============================================================

def build_excluded_dates(su_si_dates, excl_days=EXCL_DAYS):
    """
    Devuelve un pd.Index de fechas (tipo date) a excluir,
    que cubren ±excl_days alrededor de cada evento SI.
    """
    return pd.Index(
        np.unique(
            np.concatenate([
                pd.date_range(d.normalize() - pd.Timedelta(days=excl_days),
                              d.normalize() + pd.Timedelta(days=excl_days),
                              freq="D").date
                for d in pd.to_datetime(su_si_dates).dropna()
            ])
        )
    )

def build_gauge_pool(pluv_meta_csv, ruta_series, excluded_dates,
                     umbral_1d=10, umbral_7d=20, umbral_90d=80,
                     min_days_after_install=TOTAL_DAYS):
    """
    Construye:
    - cache_cs: {Codigo: cumsum_horaria}
    - candidates: {Codigo: DatetimeIndex de timestamps que cumplen los umbrales y no caen en fechas excluidas}
    """
    pl = pd.read_csv(pluv_meta_csv)
    pl["FechaInstalacion"] = pd.to_datetime(pl["FechaInstalacion"], errors="coerce")

    cache_cs = {}
    candidates = {}

    H7 = 7 * 24
    H90 = 90 * 24
    H1 = 24

    for _, row in pl.iterrows():
        cod = row["Codigo"]
        inst = row["FechaInstalacion"]

        cs = load_hourly_cumsum_for_gauge(cod, ruta_series)
        if cs is None or cs.empty:
            continue

        cache_cs[int(cod)] = cs
        idx = cs.index

        # diferencias rápidas
        ll_1d  = cs - cs.shift(H1)
        ll_7d  = cs - cs.shift(H7)
        ll_90d = cs - cs.shift(H90)

        ok = (ll_1d >= umbral_1d) & (ll_7d >= umbral_7d) & (ll_90d >= umbral_90d)

        # fuera de fechas excluidas (por día)
        ok = ok & (~pd.Index(idx.normalize().date).isin(excluded_dates))

        # respeta instalación (+ un colchón mínimo)
        if pd.notna(inst):
            ok = ok & (idx >= (inst + pd.Timedelta(days=min_days_after_install)))

        cand = idx[ok.fillna(False)]
        if len(cand) > 0:
            candidates[int(cod)] = cand

    return cache_cs, candidates



# ============================================================
# 3) Pluvio más cercano que tenga candidatos (sin depender de fecha)
# ============================================================

def assign_nearest_gauge_with_candidates(su_gdf, gpl, candidates_by_gauge, k=None):
    """
    Asigna a cada geometría en su_gdf el pluviómetro más cercano que tenga
    al menos 1 timestamp candidato.

    Nota: evitamos usar sindex.nearest(num_results=...) porque en GeoPandas
    con backend Shapely (v0.13+ / Shapely 2) ese argumento no existe. 
    Aquí, como el número de pluvios suele ser pequeño (~100-200), calculamos
    distancias directas de forma robusta.
    """
    su_gdf = su_gdf.copy()

    # solo pluvios con candidatos
    gpl = gpl[gpl["Codigo"].astype(int).isin(set(candidates_by_gauge.keys()))].copy()
    if gpl.empty:
        su_gdf["Codigo_pluvio"] = np.nan
        su_gdf["dist_pluv_m"] = np.nan
        return su_gdf

    chosen_codes = []
    chosen_dist = []

    # para acelerar un poco, preextraemos geometrías y códigos
    gpl_geom = gpl.geometry
    gpl_cod  = gpl["Codigo"].astype(int).to_numpy()

    for geom in su_gdf.geometry:
        if geom is None or geom.is_empty:
            chosen_codes.append(np.nan)
            chosen_dist.append(np.nan)
            continue

        dists = gpl_geom.distance(geom)
        j = int(dists.idxmin())  # índice del gdf (no posición)
        chosen_codes.append(int(gpl.loc[j, "Codigo"]))
        chosen_dist.append(float(dists.loc[j]))

    su_gdf["Codigo_pluvio"] = chosen_codes
    su_gdf["dist_pluv_m"] = chosen_dist
    return su_gdf


# ============================================================
# 4) CONSTRUCTOR MC (C)
# ============================================================

def construir_si_no_spatiotemporal_mc(seed,
                                      path_gpkg, layer_in,
                                      pluv_meta_csv, ruta_series, oni_path,
                                      min_dist_m=MIN_DIST_M, excl_days=EXCL_DAYS,
                                      umbral_1d=UMBRAL_1D, umbral_7d=UMBRAL_7D, umbral_90d=UMBRAL_90D,
                                      t_days=T_DAYS, p_days=P_DAYS,
                                      col_geo="geologia",
                                      col_cov="cobertura",
                                      col_slope="slope_mean",
                                      col_area="area",
                                      k_gauges=15):
    rng = np.random.default_rng(seed)

    su = gpd.read_file(path_gpkg, layer=layer_in)
    if "fid" not in su.columns:
        su = su.reset_index().rename(columns={"index": "fid"})
    
    su["fecha_hora_evento"] = pd.to_datetime(su["fecha_hora_evento"], errors="coerce")

    # CRS métrico
    if su.crs is None:
        raise ValueError("Tu layer no tiene CRS. Asigna CRS antes de medir 500m.")
    if su.crs.is_geographic:
        su = su.to_crs("EPSG:3116")
    target_crs = su.crs

    su_si = su[su["si_no"] == 1].copy()
    su_no = su[su["si_no"] == 0].copy()
    if len(su_si) == 0:
        raise ValueError("No hay filas con si_no==1 en la capa.")

    # excluir ±30 días (por día)
    excluidas = build_excluded_dates(su_si["fecha_hora_evento"], excl_days=excl_days)

    # NO elegibles por distancia (>500m)
    buffer_union = su_si.geometry.buffer(min_dist_m).unary_union
    su_no_eligible = su_no[~su_no.geometry.intersects(buffer_union)].copy()

    n_si = len(su_si)
    n_no_target = (2 * n_si)
    if len(su_no_eligible) < n_no_target:
        raise ValueError(
            f"No hay suficientes NO elegibles a >{min_dist_m}m. "
            f"Necesitas {n_no_target}, hay {len(su_no_eligible)}."
        )

    # pluvios como gdf
    gpl = read_pluvios_as_gdf(pluv_meta_csv, target_crs=target_crs)

    # pool de lluvia (cache + candidatos)
    cache_cs, candidates_by_gauge = build_gauge_pool(
        pluv_meta_csv, ruta_series, excluidas,
        umbral_1d=umbral_1d, umbral_7d=umbral_7d, umbral_90d=umbral_90d,
        min_days_after_install=T_DAYS + P_DAYS  # coherente con tu T/P
    )
    if len(candidates_by_gauge) == 0:
        raise ValueError("No se encontraron candidatos de lluvia que cumplan umbrales + exclusión. Revisa umbrales o datos.")

    # 1) muestrea SUs NO elegibles (sin reemplazo) y asigna pluvio con candidatos
    #    (oversample para no quedarte corto tras descartar los que no encuentran pluvio)
    oversample = min(len(su_no_eligible), int(n_no_target * 4))
    sampled_idx = rng.choice(su_no_eligible.index.values, size=oversample, replace=False)
    df_no = su_no_eligible.loc[sampled_idx].copy().reset_index(drop=True)

    df_no = assign_nearest_gauge_with_candidates(df_no, gpl, candidates_by_gauge, k=k_gauges)
    df_no = df_no.dropna(subset=["Codigo_pluvio"]).copy()
    if len(df_no) < n_no_target:
        raise ValueError(
            f"Tras asignar pluvios con candidatos, solo quedaron {len(df_no)}/{n_no_target} NO-eventos. "
            "Sube k_gauges, baja MIN_DIST_M o revisa umbrales/datos."
        )
    df_no = df_no.iloc[:n_no_target].copy()
    df_no["Codigo_pluvio"] = df_no["Codigo_pluvio"].astype(int)

    # 2) asigna timestamp aleatorio por cada NO-evento, desde el pool del pluvio asignado
    fechas = []
    for cod in df_no["Codigo_pluvio"].to_numpy():
        cand = candidates_by_gauge.get(int(cod), None)
        if cand is None or len(cand) == 0:
            fechas.append(pd.NaT)
        else:
            fechas.append(pd.Timestamp(rng.choice(cand.values)))  # datetime64[ns]
    df_no["fecha_hora_evento"] = pd.to_datetime(fechas, errors="coerce")
    df_no = df_no.dropna(subset=["fecha_hora_evento"]).copy()

    # 3) ONI/ENSO: merge diario (como en paquete A)
    oni = pd.read_csv(oni_path, parse_dates=["date"])
    oni["date"] = pd.to_datetime(oni["date"]).dt.normalize()

    oni_min = oni["date"].min()
    oni_max = oni["date"].max()

    # SI: filtra a rango ONI y vuelve a mergear para asegurarte consistencia
    su_si = su_si[su_si["fecha_hora_evento"].dt.normalize().between(oni_min, oni_max)].copy()
    # evita duplicados si la capa ya trae ONI/ENSO
    for col in ("ONI", "ENSO"):
        if col in su_si.columns:
            su_si = su_si.drop(columns=[col])
    su_si["date"] = su_si["fecha_hora_evento"].dt.normalize()
    su_si = su_si.merge(oni[["date", "ONI", "ENSO"]], on="date", how="left").drop(columns=["date"])

    # NO: merge
    df_no = df_no[df_no["fecha_hora_evento"].dt.normalize().between(oni_min, oni_max)].copy()
    for col in ("ONI", "ENSO"):
        if col in df_no.columns:
            df_no = df_no.drop(columns=[col])
    df_no["date"] = df_no["fecha_hora_evento"].dt.normalize()
    df_no = df_no.merge(oni[["date", "ONI", "ENSO"]], on="date", how="left").drop(columns=["date"])

    # 4) asignar pluvio válido para SI según fecha (instalación) y calcular lluvia
    #    - para NO ya tenemos Codigo_pluvio; aun así, SI necesita uno.
    su_si = su_si.copy()
    # reasignar pluvio para SI (puede tomar uno sin candidatos; eso NO importa para features)
    # elegimos el más cercano que tenga serie disponible (cache_cs)
    gpl_with_series = gpl[gpl["Codigo"].astype(int).isin(set(cache_cs.keys()))].copy()
    if gpl_with_series.empty:
        raise ValueError("No hay pluvios con series disponibles (archivos CSV) en ruta_series.")

    # asignación de pluvio para SI:
    # elegimos el más cercano (entre los k_gauges más cercanos) que cumpla instalación (>= TOTAL_DAYS)
    # Nota: evitamos sindex.nearest(num_results=...) por cambios de API en GeoPandas/Shapely. 
    su_si = su_si.reset_index(drop=True)

    codes_out = []
    dist_out = []

    gpl_series = gpl_with_series.copy()
    gpl_geom = gpl_series.geometry

    for geom, t_event in zip(su_si.geometry, su_si["fecha_hora_evento"]):
        if geom is None or geom.is_empty or pd.isna(t_event):
            codes_out.append(np.nan)
            dist_out.append(np.nan)
            continue

        dists = gpl_geom.distance(geom)

        # solo revisamos los k más cercanos por eficiencia (si k_gauges es None, revisa todos)
        if k_gauges is not None:
            idx_order = dists.nsmallest(int(k_gauges)).index
        else:
            idx_order = dists.sort_values().index

        picked = None
        picked_dist = None

        for j in idx_order:
            inst = gpl_series.loc[j, "FechaInstalacion"]
            if pd.isna(inst) or (t_event >= (inst + pd.Timedelta(days=TOTAL_DAYS))):
                picked = int(gpl_series.loc[j, "Codigo"])
                picked_dist = float(dists.loc[j])
                break

        codes_out.append(picked if picked is not None else np.nan)
        dist_out.append(picked_dist if picked_dist is not None else np.nan)

    su_si["Codigo_pluvio"] = codes_out
    su_si["dist_pluv_m"] = dist_out
    su_si = su_si.dropna(subset=["Codigo_pluvio"]).copy()
    su_si["Codigo_pluvio"] = su_si["Codigo_pluvio"].astype(int)

    # 5) lluvia para features + chequeo umbrales (NO)
    k_list = sorted(list(set([t_days, TOTAL_DAYS, 7, 90])))
    su_si = add_rain_columns_from_cache(su_si, "Codigo_pluvio", "fecha_hora_evento", cache_cs, k_list)
    df_no = add_rain_columns_from_cache(df_no, "Codigo_pluvio", "fecha_hora_evento", cache_cs, k_list)

    # refiltrar NO-eventos por umbrales (seguridad extra)
    df_no = df_no[
        (df_no["1d"] >= umbral_1d) &
        (df_no["90d"] >= umbral_90d)
    ].copy()

    # si se quedaron cortos, no seguimos (prefiero fallo explícito)
    if len(df_no) < (n_no_target - 100):
        raise ValueError(f"Tras refiltrar por umbrales quedó NO={len(df_no)}/{(n_no_target-100)}. Baja umbrales o revisa series.")

    # 6) variables temporales: Mes, DoY, sin/cos
    for dff in (su_si, df_no):
        dff["Mes"] = dff["fecha_hora_evento"].dt.month
        dff["DoY"] = dff["fecha_hora_evento"].dt.dayofyear

    enso_map = {"La Niña": 0, "Neutro": 1, "El Niño": 2}
    for dff in (su_si, df_no):
        dff["ENSO"] = dff["ENSO"].astype(str).str.strip()
        dff["ENSO"] = dff["ENSO"].replace({"Neutral": "Neutro"})
        dff["ENSO_code"] = dff["ENSO"].map(enso_map).astype("Int64")
        dff["doy_rad"] = 2 * np.pi * (dff["DoY"] - 1) / 365.0
        dff["sin_doy"] = np.sin(dff["doy_rad"])
        dff["cos_doy"] = np.cos(dff["doy_rad"])

    # 7) ensamblar + limpiar columnas necesarias 
    su_si["si_no"] = 1
    df_no["si_no"] = 0

    df_mc = pd.concat([su_si, df_no], ignore_index=True)

    needed = [
        "fecha_hora_evento", "si_no", "Codigo_pluvio",
        "ENSO_code", "sin_doy", "cos_doy",
        f"{t_days}d", f"{t_days + p_days}d",
        col_geo, col_cov, col_slope, col_area
    ]
    df_mc = df_mc.dropna(subset=needed).copy()

    df_mc["si_no"] = df_mc["si_no"].astype(int)
    df_mc["ENSO_code"] = df_mc["ENSO_code"].astype(int)

    print(
        f"[MC seed={seed}] SI={len(su_si)} | NO_eligible={len(su_no_eligible)} | "
        f"NO_final={len(df_no)} | total_final={len(df_mc)} | gauges_con_candidatos={len(candidates_by_gauge)}"
    )
    return df_mc


# ============================================================
# 5) Dataset X,y con TODAS las variables (C)
# ============================================================

def build_TP_dataset_allvars(
    df,
    t_days=T_DAYS,
    p_days=P_DAYS,
    col_geo="geologia",
    col_cov="cobertura",
    col_slope="slope_mean",
    col_area="area"
):
    total_days = t_days + p_days
    col_T = f"{t_days}d"
    col_tot = f"{total_days}d"

    needed = [
        col_T, col_tot, "sin_doy", "cos_doy", "ENSO_code",
        col_geo, col_cov, col_slope, col_area, "si_no"
    ]
    sub = df.dropna(subset=needed).copy()
    if sub.empty or sub["si_no"].nunique() < 2:
        raise ValueError("No hay datos suficientes o falta una clase.")

    # T y P
    T_vals = sub[col_T].to_numpy(dtype=float)
    P_vals = sub[col_tot].to_numpy(dtype=float) - sub[col_T].to_numpy(dtype=float)

    # temporales
    sin_doy = sub["sin_doy"].to_numpy(dtype=float)
    cos_doy = sub["cos_doy"].to_numpy(dtype=float)
    enso_cat = pd.Categorical(sub["ENSO_code"].astype(int).astype(str))  # como categórica

    # estáticas
    slope = sub[col_slope].to_numpy(dtype=float)
    area = sub[col_area].to_numpy(dtype=float)
    log_area = np.log1p(area)

    geo_cat = pd.Categorical(sub[col_geo].astype(str))
    cov_cat = pd.Categorical(sub[col_cov].astype(str))

    # --- GAM: matriz numérica + factores codificados ---
    geo_code = geo_cat.codes
    cov_code = cov_cat.codes
    enso_code = enso_cat.codes

    # orden: [T, P, sin, cos, enso_factor, slope, log_area, geo_factor, cov_factor]
    X_gam = np.column_stack([
        T_vals, P_vals, sin_doy, cos_doy,
        enso_code, slope, log_area, geo_code, cov_code
    ]).astype(float)

    # --- ML: DataFrame mixto (one-hot después) ---
    X_df_ml = pd.DataFrame({
        "T": T_vals,
        "P": P_vals,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
        "ENSO": enso_cat.astype(str),
        "slope_mean": slope,
        "log_area": log_area,
        "geologia": geo_cat.astype(str),
        "cobertura": cov_cat.astype(str),
    }, index=sub.index)

    y = sub["si_no"].to_numpy(dtype=int)
    return X_gam, X_df_ml, y


# ============================================================
# 6) Importancia por permutación (AUC drop) como en A/B
# ============================================================



FEATURE_NAMES_C = ["T", "P", "sin_doy", "cos_doy", "ENSO", "slope_mean", "log_area", "geologia", "cobertura"]

def _proba1(model, X):
    p = model.predict_proba(X)
    p = np.asarray(p)
    if p.ndim == 2:
        return p[:, 1]
    return p

def permutation_importance_auc_drop_array(model, X_test_arr, y_test, feature_names, n_perm=100, seed=0):
    rng = np.random.default_rng(seed)
    auc_base = roc_auc_score(y_test, _proba1(model, X_test_arr))

    drops_all = {}
    for j, name in enumerate(feature_names):
        drops = np.zeros(n_perm, dtype=float)
        for k in range(n_perm):
            Xp = X_test_arr.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            auc_p = roc_auc_score(y_test, _proba1(model, Xp))
            drops[k] = auc_base - auc_p
        drops_all[name] = drops

    drops_mean = {v: float(np.mean(a)) for v, a in drops_all.items()}
    drops_ci   = {v: (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) for v, a in drops_all.items()}
    return drops_mean, drops_ci, drops_all


def permutation_importance_auc_drop_df(model, X_test_df, y_test, feature_names, n_perm=100, seed=0):
    rng = np.random.default_rng(seed)
    auc_base = roc_auc_score(y_test, _proba1(model, X_test_df))

    drops_all = {}
    for name in feature_names:
        drops = np.zeros(n_perm, dtype=float)
        for k in range(n_perm):
            Xp = X_test_df.copy()
            Xp[name] = rng.permutation(Xp[name].to_numpy())
            auc_p = roc_auc_score(y_test, _proba1(model, Xp))
            drops[k] = auc_base - auc_p
        drops_all[name] = drops

    drops_mean = {v: float(np.mean(a)) for v, a in drops_all.items()}
    drops_ci   = {v: (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) for v, a in drops_all.items()}
    return drops_mean, drops_ci, drops_all


# ============================================================
# 7) MODELOS + evaluación + importancias (C)
# ============================================================

terms = (
    s(0, n_splines=6) +   # T
    s(1, n_splines=6) +   # P
    s(2, n_splines=6) +   # sin_doy
    s(3, n_splines=6) +   # cos_doy
    f(4) +                # ENSO (factor)
    s(5, n_splines=6) +   # slope_mean
    s(6, n_splines=6) +   # log_area
    f(7) +                # geologia (factor)
    f(8)                  # cobertura (factor)
)

lam_grid = np.logspace(0, 3, 5)

def get_best_lambda_train_only(X_train, y_train, random_state=42):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=random_state
    )
    best_lam = lam_grid[0]
    best_auc = -np.inf
    for lam in lam_grid:
        gam = LogisticGAM(terms, lam=lam).fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, gam.predict_proba(X_val))
        if auc > best_auc:
            best_auc = auc
            best_lam = lam
    return best_lam



def roc_and_points(y_true, y_prob, tpr_targets=(0.95,), tnr_targets=(0.95,)):
    """
    Calcula curva ROC y devuelve:
    - arrays: fpr, tpr, thr
    - puntos: OPT + puntos por objetivos de TPR y/o TNR

    Cada punto incluye threshold (prob), TPR, TNR (=1-FPR), FPR y YoudenJ.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)  # thr: de inf -> 0
    tnr = 1.0 - fpr

    # OPT = maximiza Youden J = TPR - FPR
    j = tpr - fpr
    i_opt = int(np.nanargmax(j))
    pts = [{
        "name": "OPT",
        "threshold": float(thr[i_opt]),
        "tpr": float(tpr[i_opt]),
        "tnr": float(tnr[i_opt]),
        "fpr": float(fpr[i_opt]),
        "youdenJ": float(j[i_opt]),
    }]

    # puntos por TPR objetivo
    for tt in tpr_targets:
        i = int(np.nanargmin(np.abs(tpr - tt)))
        pts.append({
            "name": f"TPR{int(round(tt*100))}",
            "threshold": float(thr[i]),
            "tpr": float(tpr[i]),
            "tnr": float(tnr[i]),
            "fpr": float(fpr[i]),
            "youdenJ": float(j[i]),
        })

    # puntos por TNR objetivo (equivale a FPR objetivo = 1 - TNR)
    for ss in tnr_targets:
        i = int(np.nanargmin(np.abs(tnr - ss)))
        pts.append({
            "name": f"TNR{int(round(ss*100))}",
            "threshold": float(thr[i]),
            "tpr": float(tpr[i]),
            "tnr": float(tnr[i]),
            "fpr": float(fpr[i]),
            "youdenJ": float(j[i]),
        })

    return fpr, tpr, thr, pts



def roc_to_records(mc_id, modelo, fpr, tpr, thr):
    # Formato "largo" (una fila por punto), fácil de guardar como CSV/Parquet
    return [
        {"mc_id": int(mc_id), "modelo": str(modelo),
         "i": int(i), "fpr": float(fpr[i]), "tpr": float(tpr[i]), "thr": float(thr[i])}
        for i in range(len(fpr))
    ]



def evaluate_models_once(df_mc,
                         t_days=T_DAYS,
                         p_days=P_DAYS,
                         test_size=0.30,
                         random_state=42,
                         thr=0.5,
                         col_geo="geologia",
                         col_cov="cobertura",
                         col_slope="slope_mean",
                         col_area="area",
                         n_perm=100,
                         seed_perm=123,
                         mc_id=None):

    X_gam, X_df_ml, y = build_TP_dataset_allvars(
        df_mc, t_days=t_days, p_days=p_days,
        col_geo=col_geo, col_cov=col_cov, col_slope=col_slope, col_area=col_area
    )

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )
    Xg_tr, Xg_te = X_gam[idx_train], X_gam[idx_test]
    Xm_tr, Xm_te = X_df_ml.iloc[idx_train].copy(), X_df_ml.iloc[idx_test].copy()
    y_tr, y_te   = y[idx_train], y[idx_test]

    def eval_once(y_true, y_prob, thr=0.5):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pofd = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        hk = (recall - pofd) if ((tp + fn) > 0 and (fp + tn) > 0) else 0.0

        return {
            "auc":  float(roc_auc_score(y_true, y_prob)),
            "ap":   float(average_precision_score(y_true, y_prob)),
            "f1":   float(f1_score(y_true, y_pred, zero_division=0)),
            "bacc": float(balanced_accuracy_score(y_true, y_pred)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "recall": float(recall),
            "precision": float(precision),
            "pofd": float(pofd),
            "hk": float(hk),
        }

    def resumen(model_name, metrics):
        return {
            "modelo": model_name,
            "n_obs": int(len(y)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "threshold": float(thr),
            **metrics
        }

    # donde acumulamos importancias
    importancias = []

    # helper para “aplanar” drops_mean/drops_ci en filas tipo tidy
    def _imp_to_records(modelo, drops_mean, drops_ci):
        recs = []
        for feat in drops_mean.keys():
            lo, hi = drops_ci[feat]
            recs.append({
                "modelo": modelo,
                "feature": feat,
                "auc_drop_mean": float(drops_mean[feat]),
                "auc_drop_lo": float(lo),
                "auc_drop_hi": float(hi),
                "n_perm": int(n_perm),
                "seed_perm": int(seed_perm),
            })
        return recs

    roc_records = []
    thr_points = []

    def add_roc(modelo, y_true, y_prob, mc_id=None):
        fpr, tpr, thr_arr, pts = roc_and_points(y_true, y_prob, tpr_targets=(0.95,), tnr_targets=(0.95,))
        roc_records.extend(roc_to_records(mc_id if mc_id is not None else -1, modelo, fpr, tpr, thr_arr))
        for p in pts:
            thr_points.append({
                "mc_id": int(mc_id if mc_id is not None else -1),
                "modelo": str(modelo),
                **p
            })
    
    # --- GAM ---
    best_lam = get_best_lambda_train_only(Xg_tr, y_tr, random_state=random_state)
    gam = LogisticGAM(terms, lam=best_lam).fit(Xg_tr, y_tr)
    met_gam = eval_once(y_te, gam.predict_proba(Xg_te), thr=thr)
    met_gam["best_lam"] = float(best_lam)

    drops_mean_g, drops_ci_g, _ = permutation_importance_auc_drop_array(
        gam, Xg_te, y_te, FEATURE_NAMES_C, n_perm=n_perm, seed=seed_perm
    )
    importancias.extend(_imp_to_records("GAM", drops_mean_g, drops_ci_g))

    # --- sklearn: columnas ---
    num_cols = ["T", "P", "sin_doy", "cos_doy", "slope_mean", "log_area"]
    cat_cols = ["ENSO", "geologia", "cobertura"]

    preproc_logit = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )
    preproc_rf = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    yprob_g = gam.predict_proba(Xg_te)
    met_gam = eval_once(y_te, yprob_g, thr=thr)
    add_roc("GAM", y_te, yprob_g, mc_id=mc_id)

    # --- Logit ---
    logit = Pipeline(steps=[
        ("prep", preproc_logit),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=2000, class_weight="balanced"
        ))
    ])
    logit.fit(Xm_tr, y_tr)
    met_logit = eval_once(y_te, logit.predict_proba(Xm_te)[:, 1], thr=thr)
    add_roc("Logit", y_te, logit.predict_proba(Xm_te)[:, 1], mc_id=mc_id)
    drops_mean_l, drops_ci_l, _ = permutation_importance_auc_drop_df(
        logit, Xm_te, y_te, FEATURE_NAMES_C, n_perm=n_perm, seed=seed_perm
    )
    importancias.extend(_imp_to_records("Logit", drops_mean_l, drops_ci_l))

    # --- RF (sin permutación aquí, como dijiste: solo GAM/Logit) ---
    rf = Pipeline(steps=[
        ("prep", preproc_rf),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            class_weight="balanced",
            random_state=0
        ))
    ])
    rf.fit(Xm_tr, y_tr)
    met_rf = eval_once(y_te, rf.predict_proba(Xm_te)[:, 1], thr=thr)
    add_roc("RandomForest", y_te, rf.predict_proba(Xm_te)[:, 1], mc_id=mc_id)

    resultados = [
        resumen("GAM", met_gam),
        resumen("Logit", met_logit),
        resumen("RandomForest", met_rf),
    ]

    models = {"gam": gam, "logit": logit, "rf": rf}
    test = {"Xg_te": Xg_te, "Xm_te": Xm_te, "y_te": y_te}

    return resultados, importancias, models, test, roc_records, thr_points



# ============================================================
# 8) MONTE CARLO + guardados (C)
# ============================================================

n_mc = 100
registros = []
import_records = []

roc_all = []
thr_all = []

# ---- AL INICIO (antes del loop) ----
no_signatures = {}   # {mc_id: set((fid, timestamp))}
no_fids_only  = {}   # {mc_id: set(fid)}  (solo espacial, sin fecha)

for mc in range(n_mc):
    seed = 100 + mc
    print(f"\n=== Monte Carlo {mc+1}/{n_mc} (seed={seed}) ===")

    try:
        df_mc = construir_si_no_spatiotemporal_mc(
            seed=seed,
            path_gpkg=path_gpkg, layer_in=layer_in,
            pluv_meta_csv=pluv_meta, ruta_series=ruta_series, oni_path=oni_path,
            min_dist_m=MIN_DIST_M, excl_days=EXCL_DAYS,
            umbral_1d=UMBRAL_1D, umbral_7d=UMBRAL_7D, umbral_90d=UMBRAL_90D,
            t_days=T_DAYS, p_days=P_DAYS,
            col_geo="geologia", col_cov="cobertura",
            col_slope="slope_mean", col_area="area"
        )

        # ---- DENTRO DEL LOOP, justo después de df_mc = construir... ----
        df_no_only = df_mc[df_mc["si_no"] == 0].copy()

        # firma espacio-temporal (fid + timestamp redondeado a hora)
        sig = set(zip(
            df_no_only["fid"].astype(int).to_numpy(),
            pd.to_datetime(df_no_only["fecha_hora_evento"]).dt.floor("H").to_numpy()
        ))

        # firma solo espacial (fid)
        sig_fid = set(df_no_only["fid"].astype(int).to_numpy())

        no_signatures[mc] = sig
        no_fids_only[mc]  = sig_fid

        print(f"[MC {mc}] NO únicos (fid,t): {len(sig)} | NO únicos fid: {len(sig_fid)}")

        resultados_mc, imps_mc, models_mc, test_mc, roc_rec, thr_pts = evaluate_models_once(
            df_mc,
            t_days=T_DAYS,
            p_days=P_DAYS,
            random_state=42,
            thr=0.5,
            n_perm=100,
            seed_perm=123,
            mc_id=mc
        )

        for r in resultados_mc:
            r["mc_id"] = mc
            registros.append(r)

        for imp in imps_mc:
            imp["mc_id"] = mc
            import_records.append(imp)

        roc_all.extend(roc_rec)
        thr_all.extend(thr_pts)

    except Exception as e:
        print(f"   -> MC {mc} falló: {e}")
        continue

results_df = pd.DataFrame(registros)
out_path = "/home/oisanchezp/Thesis/data/processed/resultados2026_paquetec_T1_P33_GAM_Logit_RF_MC100.csv"
results_df.to_csv(out_path, index=False)

imp_df = pd.DataFrame(import_records)
imp_path = "/home/oisanchezp/Thesis/data/processed/importancia2026_c_Logit_GAM.csv"
imp_df.to_csv(imp_path, index=False)

print(f"\nGuardado resultados en: {out_path}")
print(f"Guardado importancias en: {imp_path}")
print(results_df.groupby("modelo")[["auc", "ap", "f1", "bacc", "recall", "precision", "pofd", "hk"]].describe())


# ============================================================
# 9) JSON: corrida cercana a la mediana (para re-correr y efectos parciales)
# ============================================================

def mc_id_cercano_a_mediana(results_df, modelo):
    d = results_df[results_df["modelo"] == modelo].copy()
    med = float(d["auc"].median())
    idx = (d["auc"] - med).abs().idxmin()
    mc_id_star = int(results_df.loc[idx, "mc_id"])
    auc_star = float(results_df.loc[idx, "auc"])
    return med, mc_id_star, auc_star

med_gam, mc_id_star, auc_star = mc_id_cercano_a_mediana(results_df, "GAM")

selection = {
    "paquete": "C",
    "modelo_referencia": "GAM",
    "mc_id_star": mc_id_star,
    "seed_star": 100 + mc_id_star,
    "auc_star": auc_star,
    "auc_median": med_gam,
    "t_days": int(T_DAYS),
    "p_days": int(P_DAYS),
    "random_state_split": 42,
    "threshold": 0.5,
    "min_dist_m": float(MIN_DIST_M),
    "excl_days": int(EXCL_DAYS),
    "umbrales": {"1d": float(UMBRAL_1D), "7d": float(UMBRAL_7D), "90d": float(UMBRAL_90D)},
    "paths": {
        "path_gpkg": path_gpkg,
        "layer_in": layer_in,
        "pluv_meta": pluv_meta,
        "ruta_series": ruta_series,
        "oni_path": oni_path
    },
    "feature_names": FEATURE_NAMES_C
}

out_json = "/home/oisanchezp/Thesis/data/processed/seleccion_corrida_mediana_paquete_c.json"
with open(out_json, "w") as f:
    json.dump(selection, f, indent=2)

print("Guardado:", out_json)
print(selection)


roc_df = pd.DataFrame(roc_all)
roc_path = "/home/oisanchezp/Thesis/data/processed/roc_curves_paquetec_T1_P33_MC100.csv"
roc_df.to_csv(roc_path, index=False)

thr_df = pd.DataFrame(thr_all)
thr_path = "/home/oisanchezp/Thesis/data/processed/roc_threshold_points_paquetec_T1_P33_MC100.csv"
thr_df.to_csv(thr_path, index=False)

print("Guardado ROC curves:", roc_path)
print("Guardado ROC points:", thr_path)

print("\n=== FIRMAS NO-eventos por MC ===")
def jaccard(A, B):
    return len(A & B) / len(A | B) if (A or B) else np.nan

mcs = sorted(no_signatures.keys())

print("\n=== Solapamiento Jaccard entre corridas (NO espacio-temporal: fid+hora) ===")
for i in range(len(mcs)):
    for j in range(i+1, len(mcs)):
        a, b = mcs[i], mcs[j]
        J = jaccard(no_signatures[a], no_signatures[b])
        inter = len(no_signatures[a] & no_signatures[b])
        print(f"MC{a} vs MC{b}: J={J:.3f} | intersección={inter}")

print("\n=== Solapamiento Jaccard entre corridas (NO solo espacial: fid) ===")
for i in range(len(mcs)):
    for j in range(i+1, len(mcs)):
        a, b = mcs[i], mcs[j]
        J = jaccard(no_fids_only[a], no_fids_only[b])
        inter = len(no_fids_only[a] & no_fids_only[b])
        print(f"MC{a} vs MC{b}: J={J:.3f} | fids en común={inter}")
