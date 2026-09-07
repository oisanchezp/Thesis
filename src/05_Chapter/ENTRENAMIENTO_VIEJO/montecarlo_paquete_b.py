import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely 

from pygam import LogisticGAM, s, f
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score,
    balanced_accuracy_score, average_precision_score
)

# ============================================================
# 0) PARAMETROS
# ============================================================
path_gpkg = "/home/oisanchezp/Thesis/data/metadata/slope_units_con_inventario.gpkg"
layer_in  = "slope_units_full"

pluv_meta   = "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv"
ruta_series = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"

MIN_DIST_M = 500          # NO deben estar a menos de 500m de cualquier SI
T_DAYS = 1
P_DAYS = 33               # P se calcula como R_{T+P} - R_T  => total = 34
TOTAL_DAYS = T_DAYS + P_DAYS

H1 = 24                   # horas por día

# ============================================================
# 1) UTILIDADES: pluvios + lluvia
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


def assign_nearest_valid_gauge(su_gdf, gpl, min_days, date_col="fecha_hora_evento"):
    """
    Para cada SU asigna el pluviómetro más cercano que cumpla:
    fecha_evento >= FechaInstalacion + min_days
    """
    su_gdf = su_gdf.copy()
    su_gdf[date_col] = pd.to_datetime(su_gdf[date_col], errors="coerce")

    gauges_geom = gpl.geometry.values
    gauges_code = gpl["Codigo"].to_numpy()
    gauges_inst = pd.to_datetime(gpl["FechaInstalacion"]).to_numpy(dtype="datetime64[ns]")

    codes_out = np.full(len(su_gdf), np.nan)
    dist_out  = np.full(len(su_gdf), np.nan)

    delta = np.timedelta64(int(min_days), "D")

    for i, (geom, t_event) in enumerate(zip(su_gdf.geometry.values, su_gdf[date_col].to_numpy(dtype="datetime64[ns]"))):
        if geom is None or pd.isna(t_event):
            continue

        valid = t_event >= (gauges_inst + delta)
        if not np.any(valid):
            continue

        d = shapely.distance(geom, gauges_geom)  # vectorizado (Shapely 2)
        d = np.where(valid, d, np.inf)

        j = int(np.argmin(d))
        if np.isinf(d[j]):
            continue

        codes_out[i] = gauges_code[j]
        dist_out[i]  = float(d[j])

    su_gdf["Codigo_pluvio"] = codes_out
    su_gdf["dist_pluv_m"]   = dist_out
    return su_gdf



def assign_nearest_valid_gauge_sindex(su_gdf, gpl, min_days=34, date_col="fecha_hora_evento", k=15):
    """
    Para cada SU busca entre los k pluvios más cercanos el primero que cumpla:
    fecha_evento >= FechaInstalacion + min_days
    Compatible con geopandas sin max_neighbors.
    """
    su_gdf = su_gdf.copy()
    su_gdf[date_col] = pd.to_datetime(su_gdf[date_col], errors="coerce")

    # índice espacial de pluvios
    sidx = gpl.sindex

    chosen_codes = []
    chosen_dist  = []

    for geom, t_event in zip(su_gdf.geometry, su_gdf[date_col]):
        if geom is None or pd.isna(t_event):
            chosen_codes.append(np.nan)
            chosen_dist.append(np.nan)
            continue

        # candidatos: k más cercanos por bounding box (aprox) -> luego refinamos con distancia real
        # Nota: en geopandas viejos, nearest devuelve generador de índices
        cand_idx = list(sidx.nearest(geom.bounds, num_results=k))
        if len(cand_idx) == 0:
            chosen_codes.append(np.nan)
            chosen_dist.append(np.nan)
            continue

        cand = gpl.iloc[cand_idx].copy()

        # filtrar por condición de fecha
        ok = cand["FechaInstalacion"].notna() & (t_event >= (cand["FechaInstalacion"] + pd.Timedelta(days=min_days)))
        cand_ok = cand[ok].copy()

        if cand_ok.empty:
            chosen_codes.append(np.nan)
            chosen_dist.append(np.nan)
            continue

        # elegir el más cercano real (distancia geométrica)
        dists = cand_ok.geometry.distance(geom)
        j = dists.idxmin()

        chosen_codes.append(int(cand_ok.loc[j, "Codigo"]))
        chosen_dist.append(float(dists.loc[j]))

    su_gdf["Codigo_pluvio"] = chosen_codes
    su_gdf["dist_pluv_m"] = chosen_dist
    return su_gdf



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

    # encontrar índice (pad hacia atrás si no coincide exacto)
    try:
        pos = cs.index.get_loc(t)
    except KeyError:
        # usa el tiempo anterior disponible
        loc = cs.index.get_indexer([t], method="pad")
        pos = int(loc[0])
        if pos < 0:
            return np.nan

    lag = k_days * H1
    if pos - lag < 0:
        return np.nan

    return float(cs.iloc[pos] - cs.iloc[pos - lag])


def add_rain_columns(df, gauge_col, date_col, ruta_series, k_list):
    """
    Agrega columnas tipo '1d', '34d' calculadas con el pluviómetro más cercano.
    Cachea la serie por pluviómetro para no releer mil veces.
    """
    df = df.copy()
    cache = {}

    for k in k_list:
        df[f"{k}d"] = np.nan

    for cod, idxs in df.groupby(gauge_col).groups.items():
        cod = int(cod) if pd.notna(cod) else cod
        if pd.isna(cod):
            continue

        if cod not in cache:
            cache[cod] = load_hourly_cumsum_for_gauge(cod, ruta_series)

        cs = cache[cod]

        for i in idxs:
            tstamp = df.at[i, date_col]
            for k in k_list:
                df.at[i, f"{k}d"] = compute_Rk_at_time(cs, tstamp, k)

    return df


# ============================================================
# 2) CONSTRUCTOR MC: NO-eventos espaciales a >500m de SI
# ============================================================

def construir_si_no_spatial_mc(seed,
                               path_gpkg, layer_in,
                               pluv_meta_csv, ruta_series,
                               min_dist_m=500):
    rng = np.random.default_rng(seed)

    su = gpd.read_file(path_gpkg, layer=layer_in)

    # asegurar datetime
    su["fecha_hora_evento"] = pd.to_datetime(su["fecha_hora_evento"], errors="coerce")

    # CRS métricas para distancias
    if su.crs is None:
        raise ValueError("Tu layer no tiene CRS. Asigna CRS antes de medir 500m.")
    if su.crs.is_geographic:
        su = su.to_crs("EPSG:3116")  # Colombia MAGNA-SIRGAS / Colombia Bogota zone
    target_crs = su.crs

    # separar SI y NO
    su_si = su[su["si_no"] == 1].copy()
    su_no = su[su["si_no"] == 0].copy()

    if len(su_si) == 0:
        raise ValueError("No hay filas con si_no==1 en la capa.")

    # --- pool NO elegibles: distancia > min_dist_m a cualquier SI ---
    buffer_union = su_si.geometry.buffer(min_dist_m).unary_union
    su_no_eligible = su_no[~su_no.geometry.intersects(buffer_union)].copy()

    n_si = len(su_si)
    n_no_target = 2 * n_si

    if len(su_no_eligible) < n_no_target:
        raise ValueError(
            f"No hay suficientes NO elegibles a >{min_dist_m}m. "
            f"Necesitas {n_no_target}, hay {len(su_no_eligible)}."
        )

    # muestreo aleatorio de NO
    sampled_idx = rng.choice(su_no_eligible.index.values, size=n_no_target, replace=False)
    df_no = su_no_eligible.loc[sampled_idx].copy()

    # --- asignar fechas/dinámicas: duplico la lista de SI para llegar a 2*n_si ---
    si_dyn = su_si[["fecha_hora_evento", "Mes", "ONI", "ENSO"]].copy().reset_index(drop=True)
    si_dyn2 = pd.concat([si_dyn, si_dyn], ignore_index=True)

    # shuffle para no pegar siempre la misma combinación
    si_dyn2 = si_dyn2.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_no = df_no.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_no["fecha_hora_evento"] = si_dyn2["fecha_hora_evento"].values
    df_no["Mes"] = si_dyn2["Mes"].values
    df_no["ONI"] = si_dyn2["ONI"].values
    df_no["ENSO"] = si_dyn2["ENSO"].values

    # por consistencia: recalcular Mes y DoY desde fecha
    su_si["Mes"] = su_si["fecha_hora_evento"].dt.month
    su_si["DoY"] = su_si["fecha_hora_evento"].dt.dayofyear
    df_no["Mes"] = pd.to_datetime(df_no["fecha_hora_evento"]).dt.month
    df_no["DoY"] = pd.to_datetime(df_no["fecha_hora_evento"]).dt.dayofyear

    # --- pluviómetro más cercano para SI y NO ---
    gpl = read_pluvios_as_gdf(pluv_meta_csv, target_crs=su_si.crs)

    # min_days = TOTAL_DAYS = 34 para T=1 y P=33
    su_si = assign_nearest_valid_gauge(su_si, gpl, min_days=34, date_col="fecha_hora_evento")
    df_no = assign_nearest_valid_gauge(df_no, gpl, min_days=34, date_col="fecha_hora_evento")

    # elimina SUs que no encontraron un pluviómetro válido
    su_si = su_si.dropna(subset=["Codigo_pluvio"]).copy()
    df_no = df_no.dropna(subset=["Codigo_pluvio"]).copy()

    su_si["Codigo_pluvio"] = su_si["Codigo_pluvio"].astype(int)
    df_no["Codigo_pluvio"] = df_no["Codigo_pluvio"].astype(int)


    # --- lluvia: necesitamos R_1d y R_34d (total = T+P = 34) ---
    k_list = [T_DAYS, TOTAL_DAYS]  # [1, 34]
    su_si = add_rain_columns(su_si, "Codigo_pluvio", "fecha_hora_evento", ruta_series, k_list)
    df_no = add_rain_columns(df_no, "Codigo_pluvio", "fecha_hora_evento", ruta_series, k_list)

    # --- ENSO_code y cíclicas ---
    enso_map = {"La Niña": 0, "Neutro": 1, "El Niño": 2}
    for dff in [su_si, df_no]:
        dff["ENSO_code"] = dff["ENSO"].map(enso_map).astype("Int64")
        dff["doy_rad"] = 2 * np.pi * (dff["DoY"] - 1) / 365.0
        dff["sin_doy"] = np.sin(dff["doy_rad"])
        dff["cos_doy"] = np.cos(dff["doy_rad"])

    # --- ensamblar y limpiar ---
    df_combined = pd.concat([su_si, df_no], ignore_index=True)

    # quitar filas sin lluvia suficiente / sin pluvio / sin ENSO_code / sin fecha
    needed = ["fecha_hora_evento", "si_no", "Codigo_pluvio", "ENSO_code", "sin_doy", "cos_doy", "1d", f"{TOTAL_DAYS}d"]
    df_combined = df_combined.dropna(subset=needed).copy()

    # asegurar tipos
    df_combined["ENSO_code"] = df_combined["ENSO_code"].astype(int)
    df_combined["si_no"] = df_combined["si_no"].astype(int)

    print(f"[MC seed={seed}] SI={len(su_si)} | NO_elegibles={len(su_no_eligible)} | NO_sample={len(df_no)} | total_final={len(df_combined)}")
    return df_combined


# ============================================================
# 3) X,y para T=1 y P=33 usando columnas '1d' y '34d'
# ============================================================

def build_TP_dataset_newvars(
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

    needed = [col_T, col_tot, col_geo, col_cov, col_slope, col_area, "si_no"]
    sub = df.dropna(subset=needed).copy()
    if sub.empty or sub["si_no"].nunique() < 2:
        raise ValueError("No hay datos suficientes o falta una clase.")

    # --- T y P ---
    T_vals = sub[col_T].to_numpy(dtype=float)
    P_vals = sub[col_tot].to_numpy(dtype=float) - sub[col_T].to_numpy(dtype=float)

    # --- numéricas ---
    slope = sub[col_slope].to_numpy(dtype=float)

    # log-area: usa log1p para evitar problemas si hay ceros.
    # OJO: si tu area está en km^2 o m^2 da igual para el log (cambia una constante),
    # pero sé consistente y documenta unidades.
    area = sub[col_area].to_numpy(dtype=float)
    log_area = np.log1p(area)

    # --- categóricas (para GAM: códigos consistentes dentro del MC) ---
    geo_cat = pd.Categorical(sub[col_geo])
    geo_code = geo_cat.codes  # 0..K-1

    cov_cat = pd.Categorical(sub[col_cov].astype(str))  # tratar cobertura como categórica aunque sea num
    cov_code = cov_cat.codes

    # X para GAM (todo numérico + factores como int)
    X_gam = np.column_stack([T_vals, P_vals, slope, log_area, geo_code, cov_code]).astype(float)

    # X para modelos sklearn (DataFrame mixto, luego one-hot)
    X_df_ml = pd.DataFrame({
        "T": T_vals,
        "P": P_vals,
        "slope_mean": slope,
        "log_area": log_area,
        "geologia": geo_cat.astype(str),
        "cobertura": cov_cat.astype(str),
    }, index=sub.index)

    y = sub["si_no"].to_numpy(dtype=int)
    return X_gam, X_df_ml, y


# ============================================================
# 4) MODELOS (GAM + Logit + RF) + evaluación sin CV (1 split)
# ============================================================

terms = (
    s(0, n_splines=6) +   # T
    s(1, n_splines=6) +   # P
    s(2, n_splines=6) +   # slope_mean
    s(3, n_splines=6) +   # log_area
    f(4) +                # geologia (factor)
    f(5)                  # cobertura (factor)
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


def evaluate_models_once(
                        df_mc,
                        t_days=1,
                        p_days=33,
                        test_size=0.30,
                        random_state=42,
                        thr=0.5,
                        col_geo="geologia",
                        col_cov="cobertura",
                        col_slope="slope_mean",
                        col_area="area"
                        ):
    # -------- build dataset (dos vistas) --------
    X_gam, X_df_ml, y = build_TP_dataset_newvars(
        df_mc, t_days=t_days, p_days=p_days,
        col_geo=col_geo, col_cov=col_cov, col_slope=col_slope, col_area=col_area
    )

    # split consistente para ambos X (mismas filas)
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

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # POD
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pofd = fp / (fp + tn) if (fp + tn) > 0 else 0.0    # POFD
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

     # -------- Prepro para sklearn --------
    num_cols = ["T", "P", "slope_mean", "log_area"]
    cat_cols = ["geologia", "cobertura"]

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
# 5) MONTE CARLO (5 corridas)
# ============================================================

n_mc = 100
registros = []

for mc in range(n_mc):
    seed = 100 + mc
    print(f"\n=== Monte Carlo {mc+1}/{n_mc} (seed={seed}) ===")

    try:
        df_mc = construir_si_no_spatial_mc(
            seed=seed,
            path_gpkg=path_gpkg,
            layer_in=layer_in,
            pluv_meta_csv=pluv_meta,
            ruta_series=ruta_series,
            min_dist_m=MIN_DIST_M
        )

        resultados = evaluate_models_once(
            df_mc,
            t_days=1,
            p_days=33,
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
out_path = "/home/oisanchezp/Thesis/data/processed/resultados_paqueteb_3modelos_T1_P33_MC100.csv"
results_df.to_csv(out_path, index=False)

print(f"\nGuardado en: {out_path}")
print(results_df.groupby("modelo")[["auc", "ap", "f1", "bacc", "recall", "precision", "pofd", "hk"]].describe())
