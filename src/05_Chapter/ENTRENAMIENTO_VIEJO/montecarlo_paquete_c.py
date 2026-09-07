import os
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

# ============================================================
# PAQUETE C
# - NO-eventos: (lugar distinto) > 500m de cualquier SI
# - NO-eventos: (fecha distinta) fuera de ±30 días de cualquier fecha SI
# - NO-eventos: lluvia mínima: 1d>5mm, 7d>10mm, 90d>100mm
# - Variables: T, P, sin_doy, cos_doy, ENSO, geologia, cobertura, pendiente, log_area
# - Modelos: GAM + Logit + RandomForest
# ============================================================

# ============================================================
# 0) PARÁMETROS (AJUSTA A TU CASO)
# ============================================================
path_gpkg = "/home/oisanchezp/Thesis/data/metadata/slope_units_con_inventario.gpkg"
layer_in  = "slope_units_full"

pluv_meta   = "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv"
ruta_series = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"
oni_path    = "/home/oisanchezp/Thesis/data/metadata/oni_diario_2010_2025.csv"

# restricciones NO-eventos
MIN_DIST_M = 300
EXCL_DAYS  = 30

UMBRAL_1D  = 5
UMBRAL_7D  = 10
UMBRAL_90D = 80

# variables T/P (para el modelo)
T_DAYS     = 1
P_DAYS     = 33
TOTAL_DAYS = T_DAYS + P_DAYS  # 34

H1 = 24

# ============================================================
# 1) UTILIDADES: pluvios + lluvia (basado en paquete B)
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

    cs = serie_h.cumsum().astype(np.float32)
    return cs


def compute_Rk_at_time(cs, tstamp, k_days):
    """R_k (acumulado en k días) a partir de cumsum horario."""
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
# 2) CANDIDATOS: timestamps por pluvio que cumplen umbrales + exclusión ±30d
#    (idea del paquete A, pero agrupado por pluvio)
# ============================================================

def build_excluded_dates(su_si_dates, excl_days=30):
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
                     umbral_1d=5, umbral_7d=10, umbral_90d=100,
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
# 3) Asignar pluvio cercano QUE TENGA candidatos (sin necesitar fecha)
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
# 4) CONSTRUCTOR MC PAQUETE C
# ============================================================

def construir_si_no_spatiotemporal_mc(seed,
                                      path_gpkg, layer_in,
                                      pluv_meta_csv, ruta_series, oni_path,
                                      min_dist_m=500, excl_days=30,
                                      umbral_1d=5, umbral_7d=10, umbral_90d=100,
                                      t_days=1, p_days=33,
                                      col_geo="geologia",
                                      col_cov="cobertura",
                                      col_slope="slope_mean",
                                      col_area="area",
                                      k_gauges=15):
    rng = np.random.default_rng(seed)

    su = gpd.read_file(path_gpkg, layer=layer_in)
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
        min_days_after_install=t_days + p_days  # coherente con tu T/P
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
    if len(df_no) < (n_no_target - 80):
        raise ValueError(f"Tras refiltrar por umbrales quedó NO={len(df_no)}/{(n_no_target-80)}. Baja umbrales o revisa series.")

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
    t_days=1,
    p_days=33,
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
# 6) MODELOS + evaluación (igual idea que paquete B, pero con más vars)
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


def evaluate_models_once(df_mc,
                         t_days=1,
                         p_days=33,
                         test_size=0.30,
                         random_state=42,
                         thr=0.5,
                         col_geo="geologia",
                         col_cov="cobertura",
                         col_slope="slope_mean",
                         col_area="area"):

    X_gam, X_df_ml, y = build_TP_dataset_allvars(
        df_mc, t_days=t_days, p_days=p_days,
        col_geo=col_geo, col_cov=col_cov, col_slope=col_slope, col_area=col_area
    )

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )
    Xg_tr, Xg_te = X_gam[idx_train], X_gam[idx_test]
    Xm_tr, Xm_te = X_df_ml.iloc[idx_train], X_df_ml.iloc[idx_test]
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

    # --- GAM ---
    best_lam = get_best_lambda_train_only(Xg_tr, y_tr, random_state=random_state)
    gam = LogisticGAM(terms, lam=best_lam).fit(Xg_tr, y_tr)
    met_gam = eval_once(y_te, gam.predict_proba(Xg_te), thr=thr)
    met_gam["best_lam"] = float(best_lam)

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

    # --- RF ---
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

    return [
        resumen("GAM", met_gam),
        resumen("Logit", met_logit),
        resumen("RandomForest", met_rf),
    ]


# ============================================================
# 7) MONTE CARLO
# ============================================================

n_mc = 100
registros = []

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
            col_slope="slope_mean", col_area="area",
            k_gauges=15
        )

        resultados = evaluate_models_once(
            df_mc,
            t_days=T_DAYS,
            p_days=P_DAYS,
            random_state=42,
            thr=0.5
        )

        for r in resultados:
            r["mc_id"] = mc
            registros.append(r)

    except Exception as e:
        print(f"   -> MC {mc} falló: {e}")
        continue

results_df = pd.DataFrame(registros)

out_path = "/home/oisanchezp/Thesis/data/processed/resultados_paquetec_3modelos_T1_P33_MC100.csv"
results_df.to_csv(out_path, index=False)

print(f"\nGuardado en: {out_path}")
print(results_df.groupby("modelo")[["auc", "ap", "f1", "bacc", "recall", "precision", "pofd", "hk"]].describe())