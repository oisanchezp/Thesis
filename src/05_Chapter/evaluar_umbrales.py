import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask

from shapely.geometry import Polygon
from scipy.spatial.distance import pdist
from pykrige.ok import OrdinaryKriging

H1 = 24
ENSO_MAP = {"La Niña": 0, "Neutro": 1, "El Niño": 2}

# =========================
# I/O paths (tú los diste)
# =========================
PATH_GPKG  = "/home/oisanchezp/Thesis/data/metadata/slope_units_con_inventario.gpkg"
LAYER_IN   = "slope_units_full"

PLUV_META  = "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv"
RUTA_SERIES= "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"

GAM_PKL    = "/home/oisanchezp/Thesis/data/processed/gam_operativo_paquetec.pkl"
BUNDLE_JSON= "/home/oisanchezp/Thesis/data/processed/gam_bundle_paquetec.json"

# =========================
# 1) Rain helpers
# =========================
COV_NAMES = {
    0: "Cultivos",
    1: "Bosque Fragmentado",
    2: "Canteras",
    3: "Pastos",
    4: "Cuerpos de agua",
    5: "Pastos Arbolados",
    6: "Pastos Enmalezados",
    7: "Tejido Urbano Continuo",
    8: "Tejido Urbano Discontinuo",
    9: "Bosque Plantado",
}

def normalize_cobertura(s):
    # s puede venir como 1., "1.0", "Pastos", etc.
    s_num = pd.to_numeric(s, errors="coerce").astype("Int64")
    s_name = s_num.map(COV_NAMES)

    # si ya venía en texto (no numérico), lo preservamos:
    s_txt = s.astype(str).str.strip()
    out = s_name.copy()
    out[out.isna() & s.notna()] = s_txt[out.isna() & s.notna()]

    # normaliza strings raros tipo "nan"
    out = out.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return out


def load_hourly_cumsum_for_gauge(cod, ruta_series):
    fn = f"H_Datos_Procesados_Est_{cod}.csv"
    path = os.path.join(ruta_series, fn)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["Fecha","P"], parse_dates=["Fecha"])
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

def read_pluvios_as_gdf(pluv_meta_csv, target_crs):
    pl = pd.read_csv(pluv_meta_csv)
    pl["FechaInstalacion"] = pd.to_datetime(pl["FechaInstalacion"], errors="coerce")
    gpl = gpd.GeoDataFrame(
        pl,
        geometry=gpd.points_from_xy(pl["Longitude"], pl["Latitude"]),
        crs="EPSG:4326"
    ).to_crs(target_crs)
    return gpl[["Codigo","FechaInstalacion","geometry"]].copy()

def compute_gauge_accums_for_date(fecha, gpl, ruta_series, t_days, p_days):
    """
    Devuelve df con coords UTM (x,y) y acumulados:
      T = 1d, Total = 34d, P = Total - T
    """
    total_days = t_days + p_days
    rows = []
    for _, r in gpl.iterrows():
        cod = int(r["Codigo"])
        inst = r["FechaInstalacion"]

        # instalación: exigir historia suficiente
        if pd.notna(inst) and (pd.Timestamp(fecha) < inst + pd.Timedelta(days=total_days)):
            continue

        cs = load_hourly_cumsum_for_gauge(cod, ruta_series)
        if cs is None or cs.empty:
            continue

        T = compute_Rk_at_time(cs, fecha, t_days)
        Tot = compute_Rk_at_time(cs, fecha, total_days)
        if np.isnan(T) or np.isnan(Tot):
            continue

        P = Tot - T

        rows.append({
            "Codigo": cod,
            "x": float(r.geometry.x),
            "y": float(r.geometry.y),
            "T": float(T),
            "P": float(P),
            "Total": float(Tot)
        })

    return pd.DataFrame(rows)

# =========================
# 2) Kriging log-spherical (solo pluvios)
# =========================
def variogram_spherical(coords_km, z):
    maxlag = pdist(coords_km).max() / 2
    return {"nugget": 0, "sill": float(np.var(z)), "range": float(0.4 * maxlag)}

def krige_log_spherical(x_m, y_m, z, Xg_m, Yg_m, mask, scale_km=1000.0):
    z = np.asarray(z, float)
    z[z < 0] = 0
    zlog = np.log1p(z)

    coords_km = np.c_[x_m, y_m] / scale_km
    ok = OrdinaryKriging(
        coords_km[:,0], coords_km[:,1], zlog,
        variogram_model="spherical",
        variogram_parameters=variogram_spherical(coords_km, zlog),
        coordinates_type="euclidean"
    )
    zhat_log, _ = ok.execute("grid", Xg_m[0,:]/scale_km, Yg_m[:,0]/scale_km)
    zhat = np.expm1(zhat_log.filled(np.nan))
    zhat[zhat < 0] = 0
    zhat[~mask] = np.nan
    return zhat

def make_grid_from_geom(mask_geom, cell_size):
    xmin, ymin, xmax, ymax = mask_geom.bounds
    x = np.arange(xmin, xmax + cell_size, cell_size)
    y = np.arange(ymax, ymin - cell_size, -cell_size)  # <-- DESCENDENTE
    Xg, Yg = np.meshgrid(x, y)
    return Xg, Yg

def mask_polygon(geom, Xg, Yg):
    # geom: shapely (Polygon or MultiPolygon)
    from matplotlib import path
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    pts = np.c_[Xg.ravel(), Yg.ravel()]
    masks = []
    for p in polys:
        pp = path.Path(np.asarray(p.exterior.coords))
        masks.append(pp.contains_points(pts))
    m = np.any(masks, axis=0)
    return m.reshape(Xg.shape)

def write_geotiff(path, arr, transform, crs, nodata=np.nan):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32" if arr.dtype != np.int32 else "int32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)

# =========================
# 3) Zonal “modo” (con redondeo) por slope unit
# =========================

def zonal_mean_with_point_fallback(
    gdf,
    raster_path,
    min_count=1,
    all_touched=True,
    use_representative_point=True,
):
    """
    Promedio zonal por polígono. Si no hay celdas válidas (o < min_count),
    usa fallback muestreando el raster en un punto del polígono.
    
    - use_representative_point=True: usa geom.representative_point() (recomendado).
      Si False, usa centroid.
    """
    out = np.full(len(gdf), np.nan, dtype=float)

    with rasterio.open(raster_path) as src:
        nodata = src.nodata

        for i, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue

            # 1) Intento zonal
            b = geom.bounds
            win = from_bounds(*b, transform=src.transform)
            win = win.round_offsets().round_lengths()

            vals = np.array([], dtype=float)

            if win.width > 0 and win.height > 0:
                data = src.read(1, window=win, masked=True)
                if data is not None and data.size > 0:
                    win_transform = src.window_transform(win)
                    m = geometry_mask(
                        [geom],
                        invert=True,
                        out_shape=data.shape,
                        transform=win_transform,
                        all_touched=all_touched,
                    )
                    vals = np.asarray(data)[m].astype(float)
                    vals = vals[np.isfinite(vals)]
                    if nodata is not None:
                        vals = vals[vals != nodata]

            if vals.size >= min_count:
                out[i] = float(np.mean(vals))
                continue

            # 2) Fallback por punto dentro de la geometría
            pt = geom.representative_point() if use_representative_point else geom.centroid
            # sample requiere coords (x,y) en CRS del raster (aquí ya coincide)
            v = list(src.sample([(pt.x, pt.y)]))[0][0]

            if np.isfinite(v) and (nodata is None or v != nodata):
                out[i] = float(v)
            else:
                out[i] = np.nan

    return out


def zonal_mean_from_raster(gdf, raster_path, min_count=1, all_touched=True):
    """
    Promedio zonal por polígono.
    - all_touched=True: incluye píxeles tocados por el polígono (mejor para SUs pequeñas).
    - min_count: mínimo de píxeles válidos; si no, devuelve NaN.
    """
    out = np.full(len(gdf), np.nan, dtype=float)

    with rasterio.open(raster_path) as src:
        for i, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue

            b = geom.bounds
            win = from_bounds(*b, transform=src.transform)
            win = win.round_offsets().round_lengths()
            if win.width <= 0 or win.height <= 0:
                continue

            data = src.read(1, window=win, masked=True)
            if data is None:
                continue

            win_transform = src.window_transform(win)

            m = geometry_mask(
                [geom],
                invert=True,
                out_shape=data.shape,
                transform=win_transform,
                all_touched=all_touched
            )

            vals = np.asarray(data)[m]
            vals = vals[np.isfinite(vals)]
            if vals.size < min_count:
                continue

            out[i] = float(np.mean(vals))

    return out

# =========================
# 4) GAM features + predicción + alertas
# =========================
def build_X_gam_operational(su, fecha, ENSO_str, bundle):
    t_days = int(bundle["t_days"])
    p_days = int(bundle["p_days"])

    # temporales
    doy = pd.Timestamp(fecha).dayofyear
    doy_rad = 2*np.pi*(doy-1)/365.0
    sin_doy = np.sin(doy_rad)
    cos_doy = np.cos(doy_rad)

    # ENSO (factor) como code 0/1/2 -> string -> categorical con levels fijos
    enso_code_int = ENSO_MAP[ENSO_str]
    enso_cat = pd.Categorical(
        np.repeat(str(enso_code_int), len(su)),
        categories=bundle["enso_levels"]
    )
    enso_code = enso_cat.codes
    if np.any(enso_code < 0):
        raise ValueError("ENSO fuera de niveles del bundle.")

    # estáticas
    slope = su["slope_mean"].to_numpy(float)
    log_area = np.log1p(su["area"].to_numpy(float))

    # categóricas con niveles fijos
    geo_levels = bundle["geo_levels"]
    cov_levels = bundle["cov_levels"]

    geo_cat = pd.Categorical(su["geologia"].astype(str), categories=geo_levels)
    cov_cat = pd.Categorical(su["cobertura"].astype(str), categories=cov_levels)

    bad_geo = su.loc[geo_cat.codes < 0, "geologia"].astype(str).unique()
    bad_cov = su.loc[cov_cat.codes < 0, "cobertura"].astype(str).unique()
    if len(bad_geo) or len(bad_cov):
        raise ValueError(f"Categorías desconocidas. geologia={bad_geo}, cobertura={bad_cov}")

    # dinámicas ya deben estar en su["T"] y su["P"]
    T = su["T"].to_numpy(float)
    P = su["P"].to_numpy(float)

    X = np.column_stack([
        T,
        P,
        np.repeat(sin_doy, len(su)),
        np.repeat(cos_doy, len(su)),
        enso_code,
        slope,
        log_area,
        geo_cat.codes,
        cov_cat.codes
    ]).astype(float)

    return X

def classify_alerts(p, thr_prob):
    """
    thr_prob: dict con thresholds de prob: {"TPR95":.., "OPT":.., "TNR95":..}
    Retorna:
      0=Baja, 1=Media, 2=Alta, 3=Muy Alta
    """
    u_tpr95 = float(thr_prob["TPR95"])
    u_opt   = float(thr_prob["OPT"])
    u_tnr95 = float(thr_prob["TNR95"])

    # sanity (si por alguna razón no están ordenados)
    u_lo = min(u_tpr95, u_opt, u_tnr95)
    u_hi = max(u_tpr95, u_opt, u_tnr95)
    if not (u_tpr95 <= u_opt <= u_tnr95):
        # no mato el script; solo aplico orden lógico por cuantiles
        u_tpr95, u_opt, u_tnr95 = sorted([u_tpr95, u_opt, u_tnr95])

    a = np.zeros_like(p, dtype=np.int32)
    a[(p >= u_tpr95) & (p < u_opt)] = 1
    a[(p >= u_opt)   & (p < u_tnr95)] = 2
    a[p >= u_tnr95] = 3
    return a

def rasterize_from_su(su, ref_raster_path, value_col, out_path, dtype="float32", nodata=np.nan):
    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    shapes = [(geom, val) for geom, val in zip(su.geometry, su[value_col])]

    fill = nodata if dtype != "int32" else -1
    arr = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        dtype=dtype
    )
    write_geotiff(out_path, arr.astype(np.float32 if dtype=="float32" else np.int32), transform, crs, nodata=fill)

# =========================
# 5) Main operacional
# =========================
def run(fecha_str, ENSO_str, out_dir, cell_size=100, round_mode_decimals=1):
    os.makedirs(out_dir, exist_ok=True)

    # load bundle + model
    with open(BUNDLE_JSON, "r") as f:
        bundle = json.load(f)
    with open(GAM_PKL, "rb") as f:
        gam = pickle.load(f)

    t_days = int(bundle["t_days"])
    p_days = int(bundle["p_days"])
    thr_prob = bundle["thresholds_prob"]

    fecha = pd.Timestamp(fecha_str)

    # slope units
    print("Cargando slope units...")
    su = gpd.read_file(PATH_GPKG, layer=LAYER_IN)

    su["geologia"]  = su["geologia"].astype(str).str.strip()
    su["cobertura"] = normalize_cobertura(su["cobertura"])

    if "fid" not in su.columns:
        su = su.reset_index().rename(columns={"index":"fid"})
    if su.crs is None:
        raise ValueError("Slope units sin CRS.")
    if su.crs.is_geographic:
        su = su.to_crs("EPSG:3116")

    # polígono de trabajo = unión de slope units
    mask_geom = su.unary_union

    # pluvios -> CRS slope units
    print("Cargando pluvios...")
    gpl = read_pluvios_as_gdf(PLUV_META, target_crs=su.crs)
    print("n_pluvios antes filtro dominio:", len(gpl))
    # (opcional) filtrar pluvios dentro del dominio
    gpl = gpl[gpl.within(mask_geom)].copy()
    if len(gpl) < 5:
        raise ValueError("Muy pocos pluvios dentro del dominio.")

    print("n_pluvios después filtro dominio:", len(gpl))
    # acumulados por pluvio a esa fecha
    df_g = compute_gauge_accums_for_date(fecha, gpl, RUTA_SERIES, t_days=t_days, p_days=p_days)
    if len(df_g) < 10:
        raise ValueError(f"Pocos pluvios válidos para la fecha {fecha}. df_g={len(df_g)}")
    print("n_pluvios válidos para kriging:", len(df_g))

    su = su.dropna(subset=["geologia", "cobertura", "slope_mean", "area"]).copy()
    print("n_su después dropna estáticas:", len(su))
    # grid + mask
    Xg, Yg = make_grid_from_geom(mask_geom, cell_size)
    mask = mask_polygon(mask_geom, Xg, Yg)

    # transform del raster: ojo con origen (upper-left)
    xmin, ymin, xmax, ymax = mask_geom.bounds
    transform = from_origin(xmin, ymax, cell_size, cell_size)

    # kriging T y P (solo pluvios)
    zT = krige_log_spherical(df_g["x"].values, df_g["y"].values, df_g["T"].values, Xg, Yg, mask)
    zP = krige_log_spherical(df_g["x"].values, df_g["y"].values, df_g["P"].values, Xg, Yg, mask)

    # guardar rasters
    tag = fecha.strftime("%Y%m%d_%H%M")
    rT_path = os.path.join(out_dir, f"T1_{tag}.tif")
    rP_path = os.path.join(out_dir, f"P33_{tag}.tif")

    write_geotiff(rT_path, zT.astype(np.float32), transform, su.crs, nodata=np.nan)
    write_geotiff(rP_path, zP.astype(np.float32), transform, su.crs, nodata=np.nan)

    # zonal mode -> slope units
    su = su.copy()
    su["T"] = zonal_mean_with_point_fallback(su, rT_path, min_count=1, all_touched=True)
    su["P"] = zonal_mean_with_point_fallback(su, rP_path, min_count=1, all_touched=True)

    n_nan_T = su["T"].isna().sum()
    n_nan_P = su["P"].isna().sum()
    print("SUs sin T:", n_nan_T, "SUs sin P:", n_nan_P, "Total:", len(su))



    su = su.dropna(subset=["T","P","geologia","cobertura","slope_mean","area"]).copy()
    print("NaNs T:", su["T"].isna().mean(), "NaNs P:", su["P"].isna().mean())
    print("n_su después zonal (antes dropna):", len(su))
    print("T min/max:", su["T"].min(), su["T"].max())
    print("P min/max:", su["P"].min(), su["P"].max())
    # X GAM + predict
    X = build_X_gam_operational(su, fecha, ENSO_str, bundle)
    p = gam.predict_proba(X).astype(float)
    su["p_gam"] = p

    # alerts
    su["alert_code"] = classify_alerts(p, thr_prob)
    label_map = {0:"Baja", 1:"Media", 2:"Alta", 3:"Muy Alta"}
    su["alerta"] = su["alert_code"].map(label_map)

    # guardar vector
    out_gpkg = os.path.join(out_dir, f"slope_units_alertas_{tag}.gpkg")
    su = su.reset_index(drop=True)
    if "fid" in su.columns:
        su = su.drop(columns=["fid"])
    su.to_file(out_gpkg, layer="alertas", driver="GPKG")

    # raster prob y alerta (referencia = T raster)
    prob_ras = os.path.join(out_dir, f"prob_{tag}.tif")
    aler_ras = os.path.join(out_dir, f"alert_{tag}.tif")

    rasterize_from_su(su, rT_path, "p_gam", prob_ras, dtype="float32", nodata=np.nan)
    rasterize_from_su(su, rT_path, "alert_code", aler_ras, dtype="int32", nodata=-1)

    print("OK:", rT_path)
    print("OK:", rP_path)
    print("OK:", out_gpkg)
    print("OK:", prob_ras)
    print("OK:", aler_ras)
    print("Umbrales prob:", thr_prob)
    print("ENSO:", ENSO_str, "| fecha:", fecha_str, "| n_su:", len(su), "| n_pluvios:", len(df_g))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", required=True, help="YYYY-mm-dd HH:MM:SS")
    ap.add_argument("--enso", required=True, choices=["La Niña","Neutro","El Niño"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cell", type=float, default=500.0)
    ap.add_argument("--mode_dec", type=int, default=1, help="decimales para modo (redondeo)")
    args = ap.parse_args()

    run(args.fecha, args.enso, args.out_dir, cell_size=args.cell, round_mode_decimals=args.mode_dec)

if __name__ == "__main__":
    main()
