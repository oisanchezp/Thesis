"""
mapas_evento_cap5.py
====================

Mapas dinámicos hora a hora para un evento concreto — Capítulo 5.

QUÉ CAMBIA RESPECTO AL SCRIPT DEL CAPÍTULO 6
--------------------------------------------
Aquel usaba la capa de susceptibilidad (clase_tasa) y una matriz de
decisión que cruzaba susceptibilidad x nivel de lluvia. Aquí NO existe
esa capa: el nivel de alerta sale DIRECTAMENTE de la probabilidad que
predice el GAM del paquete C contra los tres cortes derivados de la
curva ROC. Es el esquema de cuatro clases de la Tabla 5.3:

    p <  P*(TPR95)                  -> Low
    P*(TPR95) <= p < P*(TPR85)      -> Medium
    P*(TPR85) <= p < P*(TPR70)      -> High
    p >= P*(TPR70)                  -> Very high

El modelo lleva las siete variables del Capítulo 5 (STR, LTR, pendiente,
curvatura, cobertura, suelos y fase ENSO), no las cuatro del pooled.

SALIDAS
-------
    1. fig_event_<tag>_alert.pdf/.png
       Rejilla de mapas del zoom a distintas horas: evolución de la alerta.
    2. fig_event_<tag>_rain.pdf/.png
       La misma rejilla pero coloreada por lluvia de corto plazo (STR),
       para ver cómo evoluciona la lluvia que produce esa alerta.
    3. frames/frame_###.png  +  <tag>.gif
       Los fotogramas de la animación: valle completo + zoom + hietograma.

VALIDACIÓN INDEPENDIENTE
------------------------
Los dos eventos configurados (Olivares, 5 de mayo; Granizal, 24 de junio
de 2025) ocurrieron DESPUÉS del hold-out del 1 de abril de 2025, así que
el modelo no vio esa lluvia ni esas laderas durante el entrenamiento.

USO
---
    python mapas_evento_cap5.py            # corre el evento de EVENTO_ACTIVO
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import ListedColormap, BoundaryNorm

from mc_common import CFG, _proba1, asignar_su_uid

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE        = "/home/oisanchezp/Thesis"
MC_DIR      = f"{BASE}/src/05_Chapter/Montecarlo/"
PLUV_META   = f"{BASE}/data/metadata/pluviometros_metadatos_20250506.csv"
RUTA_SERIES = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"
ONI_PATH    = f"{BASE}/data/metadata/oni_diario_2010_2025.csv"
MUNI_SHP    = f"{BASE}/data/metadata/Valle_Aburra_Def_con_id.shp"
RIO_GPKG    = f"{BASE}/data/metadata/rio_aburra.gpkg"
OUT_DIR     = f"{BASE}/src/05_Chapter/Mapas/"

# --- eventos disponibles ---------------------------------------------
EVENTOS = {
    "olivares": dict(
        nombre      = "Olivares",
        csv         = f"{BASE}/data/altavista/eventos_olivares_20250505.csv",
        frame_start = pd.Timestamp("2025-05-04 19:00"),
        frame_end   = pd.Timestamp("2025-05-05 07:00"),
    ),
    "granizal": dict(
        nombre      = "Granizal",
        csv         = f"{BASE}/data/altavista/eventos_granizal_20250624.csv",
        frame_start = pd.Timestamp("2025-06-23 19:00"),
        frame_end   = pd.Timestamp("2025-06-24 06:00"),
    ),
    "altavista": dict(
    nombre      = "Altavista",
    csv         = f"{BASE}/data/altavista/eventos_altavista_20250428.csv",
    frame_start = pd.Timestamp("2025-04-27 18:00"),
    frame_end   = pd.Timestamp("2025-04-28 04:00"),
    )
}
EVENTO_ACTIVO = "granizal"

FRAME_HOURS   = 1        # un fotograma por hora
SEG_POR_FRAME = 0.30     # duración de cada fotograma en el GIF
ZOOM_MARGEN_M = 1500     # margen del zoom alrededor de los deslizamientos

# Recuadro de zoom dentro de cada panel: [x0, y0, ancho, alto] en fracción
# de los ejes. Por defecto, esquina inferior derecha.
INSET_POS = [0.56, 0.03, 0.42, 0.42]
# Conexiones visibles entre el recuadro del mapa grande y el zoom.
# Orden: 0 = inferior-izquierda, 1 = superior-izquierda,
#        2 = inferior-derecha,   3 = superior-derecha.
# Con el zoom abajo a la derecha y Altavista al occidente del valle, las
# dos de la izquierda son las que no cruzan el mapa.
INSET_CONECTORES = (0, 1)

# Horas que se muestran en las rejillas de paneles. None = 6 repartidas
# de forma uniforme entre frame_start y frame_end.
HORAS_PANEL = None
N_PANELES   = 6          # 4 o 6

# CRS métrico del proyecto. Todo cálculo de distancias (pluviómetro más
# cercano, buffers, zoom) va en metros; EPSG:4326 solo para dibujar.
CRS_METRICO = "EPSG:3116"

# --- niveles de alerta ------------------------------------------------
# El corte MAS PERMISIVO (TPR95) produce el nivel MAS BAJO de alerta.
CLAVES_CORTE   = ["P_TPR95", "P_TPR85", "P_TPR70"]
NOMBRES_ALERTA = ["Low", "Medium", "High", "Very high"]
COLORES_ALERTA = ["#2E7D32", "#F9A825", "#EF6C00", "#C62828"]

# Unidades que el modelo no puede evaluar (estáticas vacías o clase de
# cobertura/suelo no vista en el entrenamiento). Se dibujan en gris.
NO_EVALUABLE   = -1
COLOR_NO_EVAL  = "#BDBDBD"
COLORES_MAPA   = [COLOR_NO_EVAL] + COLORES_ALERTA
LIMITES_MAPA   = [-1.5, -.5, .5, 1.5, 2.5, 3.5]

COLOR_LINDE = "#37474F"
COLOR_RIO   = "#0D47A1"
COLOR_INT   = "#0277BD"
COLOR_ACUM  = "#263238"

# Variables del modelo del Capítulo 5
NUM  = ["T", "P", "slope_mean", "curva_mean"]
CAT  = ["cobertura", "suelos", "ENSO"]
FEAT = NUM + CAT


# ============================================================
# LLUVIA
# ============================================================

def load_hourly_cumsum(cod: int) -> Optional[pd.Series]:
    """Serie horaria acumulada de un pluviómetro.

    Duplicados con max() y huecos como NaN, igual que en mc_common: si
    aquí se usara sum() o fill_value=0 la lluvia no coincidiría con la
    del entrenamiento y el mapa sería incoherente con los umbrales.
    """
    path = os.path.join(RUTA_SERIES, f"H_Datos_Procesados_Est_{cod}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["Fecha", "P"], parse_dates=["Fecha"])
    df = df.dropna(subset=["Fecha"])
    serie = df.groupby("Fecha")["P"].max().sort_index().asfreq("H")
    return serie.fillna(0.0).cumsum().astype(np.float32)


def acumulado(cs: pd.Series, ts: pd.Timestamp, horas: int) -> float:
    """Lluvia acumulada en las `horas` previas a ts. NaN fuera del rango."""
    if cs is None or cs.empty:
        return np.nan
    t = pd.Timestamp(ts).floor("H")
    if t < cs.index[0] or t > cs.index[-1]:
        return np.nan
    try:
        pos = cs.index.get_loc(t)
    except KeyError:
        pos = int(cs.index.get_indexer([t], method="pad")[0])
        if pos < 0:
            return np.nan
    if pos - horas < 0:
        return np.nan
    return float(cs.iloc[pos] - cs.iloc[pos - horas])


# ============================================================
# ESTILO DE LOS MAPAS
# ============================================================

def fmt_lon(x, _):
    return f"{abs(x):.2f}° {'W' if x < 0 else 'E'}"


def fmt_lat(y, _):
    return f"{abs(y):.2f}° {'S' if y < 0 else 'N'}"


def estilo_mapa(ax, bounds=None, coords=True, nx=3, ny=4, marco=True):
    """Marco fino, marcas hacia ADENTRO y coordenadas solo abajo y a la
    izquierda. Sin títulos de eje: las unidades se leen en las etiquetas.

    Con `marco=False` se ocultan los cuatro lados y quedan solo las marcas
    de abajo y de la izquierda con sus coordenadas. Es lo que conviene
    cuando el mapa no es rectangular y el marco encierra mucho vacío.
    """
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    for sp in ax.spines.values():
        sp.set_visible(marco)
        sp.set_linewidth(0.9)
        sp.set_edgecolor("black")
    if coords:
        ax.tick_params(direction="in", length=4.5, width=0.9,
                       top=marco, right=marco,
                       labeltop=False, labelright=False,
                       labelbottom=True, labelleft=True, labelsize=7.5)
        ax.xaxis.set_major_locator(MaxNLocator(nx, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(ny, prune="both"))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")


# ============================================================
# PREPARACIÓN
# ============================================================

def ajustar_o_cargar_gam():
    """Reajusta el GAM de la corrida representativa del paquete C, o lo
    lee de caché. Reajustar tarda minutos, así que se guarda."""
    cache_path = os.path.join(MC_DIR, "gam_set_C_star.joblib")
    if os.path.exists(cache_path):
        print("GAM cargado de caché:", cache_path)
        return joblib.load(cache_path)

    print("Reajustando el GAM de la corrida representativa (tarda)...")
    from sklearn.model_selection import GroupShuffleSplit
    from mc_common import make_models
    import mc_set_c

    met = pd.read_csv(os.path.join(MC_DIR, "set_C_mc_metrics.csv"))
    met = met[met.model == "GAM"]
    med = met["auc"].median()
    run_star = int(met.loc[(met["auc"] - med).abs().idxmin(), "run_id"])
    seed = CFG["BASE_SEED"] + run_star

    df = mc_set_c.build_dataset(seed)
    df = df.dropna(subset=["si_no"] + FEAT).copy().reset_index(drop=True)
    y = df["si_no"].astype(int).to_numpy()
    Xdf = df[FEAT].copy()
    for c in CAT:
        Xdf[c] = Xdf[c].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=CFG["TEST_SIZE"],
                            random_state=seed)
    itr, _ = next(gss.split(Xdf, y, groups=df["su_uid"].to_numpy()))

    gam = make_models(NUM, CAT, seed)["GAM"]
    gam.fit(Xdf.iloc[itr], y[itr], seed=seed)
    joblib.dump(gam, cache_path)
    print(f"GAM ajustado (run {run_star}) y guardado en {cache_path}")
    return gam


def leer_cortes():
    """Mediana de los tres cortes sobre las corridas Monte Carlo."""
    thr = pd.read_csv(os.path.join(MC_DIR, "set_C_mc_thresholds.csv"))
    thr = thr[thr.model == "GAM"]
    cortes = thr.groupby("thr_level")["threshold"].median()
    p = [float(cortes[k]) for k in CLAVES_CORTE]
    print(f"Cortes p*: TPR95={p[0]:.3f} | TPR85={p[1]:.3f} | TPR70={p[2]:.3f}")
    return p


def cargar_capas_contexto(crs):
    """Contornos de municipios, comunas de Medellín y río Aburrá."""
    ver = gpd.read_file(MUNI_SHP).to_crs(crs)
    ver["geometry"] = ver.geometry.buffer(0)
    municipios = ver.dissolve("Municipio").boundary
    es_med  = ver["Municipio"].astype(str).str.upper().str.contains("MEDELL")
    comunas = ver[es_med].dissolve("Comuna").boundary
    rio = gpd.read_file(RIO_GPKG).to_crs(crs)
    return {"municipios": municipios, "comunas": comunas, "rio": rio}


def marcar_evaluables(su, gam):
    """Marca qué slope units puede evaluar el modelo, y por qué no las demás.

    El GAM se entrenó con 1.170 filas. Ese subconjunto no contiene todas las
    clases de cobertura y suelo que existen en las 61.073 unidades del valle,
    y algunas unidades tienen estáticas vacías. Al predecir, una clase no vista
    se convierte en NaN dentro del codificador y pyGAM aborta.

    La salida honesta no es imputar sino declarar: esas unidades quedan como
    'no evaluables' y se dibujan en gris. Si son pocas, es una nota al pie;
    si son muchas, es una limitación que hay que escribir en la Discussion.
    """
    ok = pd.Series(True, index=su.index)

    for c in CFG["STATIC_NUM"]:
        falta = pd.to_numeric(su[c], errors="coerce").isna()
        if falta.any():
            print(f"  {c}: {int(falta.sum())} unidades sin valor")
        ok &= ~falta

    # Categorías que el GAM nunca vio. cat_maps se llena en GamWrapper.fit
    for c in CFG["STATIC_CAT"]:
        vistas = set(getattr(gam, "cat_maps", {}).get(c, {}).keys())
        if not vistas:
            continue
        val = su[c].astype(str)
        nueva = ~val.isin(vistas)
        if nueva.any():
            print(f"  {c}: {int(nueva.sum())} unidades con clase no vista "
                  f"en el entrenamiento -> {sorted(val[nueva].unique())[:6]}")
        ok &= ~nueva

    su = su.copy()
    su["evaluable"] = ok.to_numpy()
    n_no = int((~ok).sum())
    print(f"Unidades evaluables: {int(ok.sum())} de {len(su)} "
          f"({100 * n_no / len(su):.1f} % sin evaluar)")
    return su


def preparar(ev_cfg):
    # --- slope units del Capítulo 5 -----------------------------------
    su = gpd.read_file(CFG["PATH_SU_GPKG"], layer=CFG["LAYER"])
    if su.crs is None:
        raise ValueError("El gpkg no tiene CRS.")
    # El gpkg no trae `su_uid`: se resuelve con el mismo criterio que usan
    # los paquetes de entrenamiento, para que las unidades del mapa sean
    # las mismas que las del modelo.
    su = asignar_su_uid(su, avisar=False)
    # Una fila por slope unit: aquí no interesan las fechas de evento, solo
    # la geometría y las estáticas, y el join las duplicó.
    su = su.drop_duplicates(subset="su_uid").reset_index(drop=True)
    # El emparejamiento con el pluviómetro se hace en METROS (EPSG:3116),
    # igual que en el entrenamiento. Hacerlo en grados daría un vecino
    # distinto al que usó el modelo, porque un grado de longitud y uno de
    # latitud no miden lo mismo. Solo al final se pasa a EPSG:4326 para
    # dibujar con coordenadas geográficas.
    su_m = su.to_crs(CRS_METRICO) if su.crs.is_geographic else su.copy()
    print(f"Slope units: {len(su_m)}")

    gam = ajustar_o_cargar_gam()
    p_cortes = leer_cortes()

    # --- pluviómetros con serie completa en la ventana ----------------
    pl = pd.read_csv(PLUV_META)
    gpl = gpd.GeoDataFrame(
        pl, geometry=gpd.points_from_xy(pl["Longitude"], pl["Latitude"]),
        crs="EPSG:4326").to_crs(CRS_METRICO)
    gpl["Codigo"] = gpl["Codigo"].astype(int)

    ini_req = ev_cfg["frame_start"] - pd.Timedelta(days=31)
    cache = {}
    for cod in gpl["Codigo"].unique():
        cs = load_hourly_cumsum(int(cod))
        if cs is None or cs.empty:
            continue
        if cs.index[0] <= ini_req and cs.index[-1] >= ev_cfg["frame_end"]:
            cache[int(cod)] = cs
    gpl = gpl[gpl["Codigo"].isin(cache)].copy()
    print(f"Pluviómetros con serie completa: {len(gpl)}")
    if gpl.empty:
        raise RuntimeError("Ningún pluviómetro cubre la ventana del evento")

    # --- pluviómetro más cercano por slope unit (en metros) -----------
    cent = gpd.GeoDataFrame(geometry=su_m.geometry.centroid, crs=su_m.crs)
    j = gpd.sjoin_nearest(cent, gpl[["Codigo", "geometry"]], how="left",
                          distance_col="dist_pluv")
    j = j[~j.index.duplicated(keep="first")]
    su_m["Codigo_pluvio"] = j["Codigo"].astype(int).to_numpy()
    su_m["dist_pluv_km"] = j["dist_pluv"].to_numpy() / 1000.0
    print(f"Distancia al pluviómetro: mediana={su_m['dist_pluv_km'].median():.1f} km"
          f" | p90={su_m['dist_pluv_km'].quantile(.9):.1f} km"
          f" | max={su_m['dist_pluv_km'].max():.1f} km")

    # --- unidades evaluables por el modelo ----------------------------
    # Una slope unit solo se puede evaluar si sus estáticas están completas
    # y sus clases de cobertura y suelo estaban en el entrenamiento. Las
    # demás se marcan y se dibujan en gris: imputarlas sería inventar.
    su_m = marcar_evaluables(su_m, gam)

    # --- a coordenadas geográficas, solo para dibujar -----------------
    su = su_m.to_crs("EPSG:4326")
    su["geometry"] = su.geometry.simplify(0.0002)   # acelera el render

    # --- ONI -> fase ENSO ---------------------------------------------
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    oni["ENSO"] = (oni["ENSO"].astype(str).str.strip()
                   .replace({"Neutral": "Neutro"}))
    oni = oni.set_index(oni["date"].dt.normalize())["ENSO"]
    oni = oni[~oni.index.duplicated()]

    # --- deslizamientos observados del evento -------------------------
    ev = pd.read_csv(ev_cfg["csv"], parse_dates=["fecha_hora"])
    ev = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["lon"], ev["lat"]),
                          crs="EPSG:4326").to_crs(su.crs)
    print(f"Deslizamientos del evento: {len(ev)}")

    # --- malla de interpolación, si los mapas van con campo kriging ---
    if METODO_LLUVIA == "kriging":
        _INTERP.update(preparar_interpolacion(su_m, gpl))

    gpl = gpl.to_crs("EPSG:4326")     # ya se usó en metros; ahora, dibujar
    capas = cargar_capas_contexto(su.crs)
    return su, gam, p_cortes, cache, oni, ev, gpl, capas


# ============================================================
# LLUVIA INTERPOLADA (solo para evaluar y mostrar los eventos)
# ============================================================
# El entrenamiento, la selección de ventanas y los cortes de probabilidad
# usan la lluvia del PLUVIÓMETRO MÁS CERCANO, y eso no cambia. Aquí se
# ofrece además un campo interpolado para los mapas del evento, porque el
# vecino más cercano produce polígonos de Thiessen con saltos artificiales
# que no existen en la lluvia real.
#
# La sustitución no es gratuita. El kriging suaviza: baja los picos y sube
# los valles, y el promedio por slope unit suaviza una segunda vez. Sobre
# este evento el campo resultó demasiado plano frente al que vio el modelo,
# así que el valor por defecto es "vecino": el mapa muestra la misma lluvia
# con la que se calibraron los cortes, aunque se vean los bordes de las
# áreas de influencia. `comparar_metodos()` sigue disponible para medir la
# diferencia y poder citarla.

METODO_LLUVIA = "vecino"     # "vecino" (consistente) | "kriging" (suavizado)
GRID_M        = 250.0        # resolución de la malla de interpolación, en m
VARIOGRAMA    = "spherical"

_INTERP = {}                 # lo llena preparar(); evita cambiar firmas
_CACHE_CAMPO = {}            # (ts, horas) -> mm por slope unit


def _krige_log(x, y, z, xg, yg):
    """Kriging ordinario sobre log(1+z), como en el modelo operacional.

    Se interpola en logaritmo porque la lluvia es positiva y muy asimétrica;
    hacerlo en crudo genera valores negativos alrededor de los ceros.
    """
    from pykrige.ok import OrdinaryKriging
    from scipy.spatial.distance import pdist

    z = np.asarray(z, float).copy()
    z[z < 0] = 0.0
    zl = np.log1p(z)

    c_km = np.c_[x, y] / 1000.0
    maxlag = float(pdist(c_km).max()) / 2.0
    ok = OrdinaryKriging(
        c_km[:, 0], c_km[:, 1], zl,
        variogram_model=VARIOGRAMA,
        variogram_parameters={"nugget": 0.0,
                              "sill": float(np.var(zl)) or 1e-6,
                              "range": 0.4 * maxlag},
        coordinates_type="euclidean")
    zh, _ = ok.execute("grid", xg / 1000.0, yg / 1000.0)
    out = np.expm1(np.asarray(zh, float))
    out[~np.isfinite(out)] = np.nan
    out[out < 0] = 0.0
    return out                      # (ny, nx)


def preparar_interpolacion(su_m, gpl_m):
    """Precalcula la malla y a qué slope unit pertenece cada celda.

    El campo de lluvia cambia con la hora, pero la geometría no, así que
    esta correspondencia se calcula una sola vez.
    """
    x0, y0, x1, y1 = su_m.total_bounds
    xg = np.arange(x0, x1 + GRID_M, GRID_M)
    yg = np.arange(y0, y1 + GRID_M, GRID_M)
    XX, YY = np.meshgrid(xg, yg)

    base = su_m[["geometry"]].copy()
    base["_pos"] = np.arange(len(su_m))
    pts = gpd.GeoDataFrame(
        {"_celda": np.arange(XX.size)},
        geometry=gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=su_m.crs)
    j = gpd.sjoin(pts, base, how="inner", predicate="within")

    cent = su_m.geometry.centroid
    print(f"Malla de interpolación: {len(xg)}x{len(yg)} celdas de {GRID_M:.0f} m "
          f"| {len(j)} celdas dentro de alguna slope unit")

    return {"xg": xg, "yg": yg,
            "celda": j["_celda"].to_numpy(),
            "pos":   j["_pos"].to_numpy(),
            "n_su":  len(su_m),
            "cx":    cent.x.to_numpy(), "cy": cent.y.to_numpy(),
            "gpl":   gpl_m}


def lluvia_interpolada(ts, horas, cache):
    """Promedio por slope unit del campo interpolado de lluvia acumulada."""
    clave = (pd.Timestamp(ts), int(horas))
    if clave in _CACHE_CAMPO:
        return _CACHE_CAMPO[clave]

    z_ = _INTERP
    gpl_m = z_["gpl"]
    z = np.array([acumulado(cache[int(c)], ts, horas) if int(c) in cache
                  else np.nan for c in gpl_m["Codigo"]], float)
    ok = np.isfinite(z)
    if ok.sum() < 5:
        raise RuntimeError(f"Solo {int(ok.sum())} pluviómetros válidos en {ts}")

    campo = _krige_log(gpl_m.geometry.x.to_numpy()[ok],
                       gpl_m.geometry.y.to_numpy()[ok], z[ok],
                       z_["xg"], z_["yg"])

    plano = campo.ravel()
    val = pd.Series(plano[z_["celda"]]).groupby(z_["pos"]).mean()

    out = np.full(z_["n_su"], np.nan)
    out[val.index.to_numpy()] = val.to_numpy()

    # Unidades más pequeñas que una celda: no contienen ningún punto de la
    # malla, así que se muestrea el campo en su centroide.
    faltan = ~np.isfinite(out)
    if faltan.any():
        ix = np.clip(np.searchsorted(z_["xg"], z_["cx"][faltan]),
                     0, len(z_["xg"]) - 1)
        iy = np.clip(np.searchsorted(z_["yg"], z_["cy"][faltan]),
                     0, len(z_["yg"]) - 1)
        out[faltan] = campo[iy, ix]

    _CACHE_CAMPO[clave] = out
    return out


def comparar_metodos(ts, su, cache, p_cortes):
    """Cuantifica la diferencia entre el vecino más cercano y el campo
    interpolado. Este es el número que justifica usar el segundo en los
    mapas mientras el modelo sigue calibrado con el primero."""
    r1 = {c: acumulado(cs, ts, 24) for c, cs in cache.items()}
    nn = su["Codigo_pluvio"].map(r1).to_numpy(float)
    kr = lluvia_interpolada(ts, 24, cache)

    d = pd.DataFrame({"vecino": nn, "kriging": kr}).dropna()
    dif = d["kriging"] - d["vecino"]
    print(f"\n--- STR de 24 h en {ts:%d %b %H:%M} ---")
    print(d.describe().round(1).to_string())
    print(f"Sesgo medio : {dif.mean():+.1f} mm")
    print(f"RMSE        : {np.sqrt((dif ** 2).mean()):.1f} mm")
    print(f"Correlación : {d.corr().iloc[0, 1]:.3f}")
    print(f"p99 vecino  : {d['vecino'].quantile(.99):.1f} mm | "
          f"kriging: {d['kriging'].quantile(.99):.1f} mm")
    for u in (CFG["UMBRAL_1D"], 20.0, 30.0, 40.0):
        print(f"  SU sobre {u:>4.0f} mm -> vecino {int((d.vecino > u).sum()):>6} "
              f"| kriging {int((d.kriging > u).sum()):>6}")
    return d


# ============================================================
# MOTOR: nivel de alerta por slope unit en un instante
# ============================================================

def evaluar_valle(ts, su, gam, p_cortes, cache, oni):
    """Nivel de alerta 0..3 por slope unit, más STR y LTR.

    A diferencia del Capítulo 6, no hay matriz de decisión: el nivel sale
    directamente de comparar la probabilidad con los tres cortes.
    """
    ts = pd.Timestamp(ts)

    if METODO_LLUVIA == "kriging":
        R1  = pd.Series(lluvia_interpolada(ts, 24, cache), index=su.index)
        R30 = pd.Series(lluvia_interpolada(ts, 30 * 24, cache), index=su.index)
    else:
        r1, r30 = {}, {}
        for cod, cs in cache.items():
            r1[cod]  = acumulado(cs, ts, 24)
            r30[cod] = acumulado(cs, ts, 30 * 24)
        R1  = su["Codigo_pluvio"].map(r1)
        R30 = su["Codigo_pluvio"].map(r30)

    enso = oni.get(ts.normalize(), None)
    if enso is None:
        raise ValueError(f"Sin fase ENSO para {ts}")

    # Dos condiciones distintas, que conviene no confundir:
    #   - evaluable: el modelo PUEDE dar una probabilidad (estáticas completas
    #     y clases vistas en el entrenamiento). Si no, nivel = -1 (gris).
    #   - activa   : además supera el filtro de exposición 10 mm / 50 mm, el
    #     mismo del entrenamiento. Si no, nivel = 0 (sin alerta).
    evaluable = su["evaluable"] if "evaluable" in su.columns \
        else pd.Series(True, index=su.index)

    activa = (evaluable
              & (R1 > CFG["UMBRAL_1D"]) & (R30 > CFG["UMBRAL_30D"])
              & R1.notna() & R30.notna())

    nivel = np.zeros(len(su), dtype=int)
    nivel[~evaluable.to_numpy()] = NO_EVALUABLE
    if activa.any():
        d = su.loc[activa]
        grid = pd.DataFrame({
            "T": R1[activa].to_numpy(float),
            "P": (R30[activa] - R1[activa]).clip(lower=0).to_numpy(float),
            "slope_mean": d["slope_mean"].astype(float).to_numpy(),
            "curva_mean": d["curva_mean"].astype(float).to_numpy(),
            "cobertura":  d["cobertura"].astype(str).to_numpy(),
            "suelos":     d["suelos"].astype(str).to_numpy(),
            "ENSO":       str(enso),
        })
        # Red de seguridad: si algo se coló como NaN, el mensaje debe decir
        # QUÉ columna falla, no el críptico "X must not contain Inf nor NaN".
        malas = grid[FEAT].isna().sum()
        if malas.any():
            raise ValueError(
                f"NaN en el grid de {ts}: "
                f"{malas[malas > 0].to_dict()}. Revisa marcar_evaluables().")
        p = _proba1(gam, grid[FEAT])
        n = np.zeros(len(p), dtype=int)
        n[p >= p_cortes[0]] = 1
        n[p >= p_cortes[1]] = 2
        n[p >= p_cortes[2]] = 3
        nivel[activa.to_numpy()] = n

    return pd.Series(nivel, index=su.index), R1, R30


# ============================================================
# FIGURA 1 y 2 — rejillas de paneles del zoom
# ============================================================

def _pintar(ax, su, nivel, R1, ocurridos, capas, modo,
            cmap_al, norm_al, vmax, ms_ev=26, lw=1.0):
    """Dibuja el contenido del mapa. Se llama dos veces por panel, una para
    el valle completo y otra para el zoom, de modo que ambos muestren
    exactamente lo mismo y solo cambie la extensión."""
    if modo == "alerta":
        su.assign(v=nivel.to_numpy()).plot(
            ax=ax, column="v", cmap=cmap_al, norm=norm_al, linewidth=0)
    else:
        su.assign(v=R1.to_numpy()).plot(
            ax=ax, column="v", cmap="YlGnBu", vmin=0, vmax=vmax,
            linewidth=0, missing_kwds={"color": "#eeeeee"})

    capas["rio"].plot(ax=ax, color=COLOR_RIO, linewidth=lw, zorder=6)
    # Comunas primero y municipios encima: los límites municipales son la
    # referencia territorial y deben leerse por encima de la subdivisión.
    capas["comunas"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.45 * lw,
                          alpha=0.75, zorder=7)
    capas["municipios"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.95 * lw,
                             zorder=8)
    if len(ocurridos):
        ocurridos.plot(ax=ax, color="black", edgecolor="white",
                       markersize=ms_ev, linewidth=0.5, zorder=10)


def rejilla_paneles(horas, su, gam, p_cortes, cache, oni, ev, capas, zb,
                    modo, tag, nombre):
    """modo='alerta' -> nivel de alerta ; modo='lluvia' -> STR en mm.

    Cada panel lleva el Valle de Aburrá completo y, en la esquina inferior
    derecha, un zoom de la zona afectada. El recuadro sobre el mapa grande
    y las líneas de conexión indican de dónde sale el zoom.
    """
    n = len(horas)
    ncol = 3 if n % 3 == 0 else 2
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.9 * nrow),
                             squeeze=False, sharex=True, sharey=True)

    cmap_al = ListedColormap(COLORES_MAPA)
    norm_al = BoundaryNorm(LIMITES_MAPA, cmap_al.N)

    vb = tuple(su.total_bounds)          # extensión del valle completo

    # escala común de lluvia entre paneles: se calcula antes de dibujar
    vmax = 1.0
    if modo == "lluvia":
        for ts in horas:
            _, R1, _ = evaluar_valle(ts, su, gam, p_cortes, cache, oni)
            vmax = max(vmax, float(np.nanpercentile(R1.dropna(), 99)))

    for k, ts in enumerate(horas):
        ax = axes[k // ncol][k % ncol]
        nivel, R1, _ = evaluar_valle(ts, su, gam, p_cortes, cache, oni)
        ocurridos = ev[ev["fecha_hora"] <= ts]

        # --- mapa principal: todo el valle ---------------------------
        _pintar(ax, su, nivel, R1, ocurridos, capas, modo,
                cmap_al, norm_al, vmax, ms_ev=7, lw=0.7)
        estilo_mapa(ax, vb, coords=True, nx=3, ny=4)
        ax.set_aspect("equal")
        ax.set_title(f"({'abcdefgh'[k]})  {ts:%d %b %H:%M}", loc="left",
                     fontsize=10)

        # --- zoom de la zona afectada --------------------------------
        axz = ax.inset_axes(INSET_POS)
        _pintar(axz, su, nivel, R1, ocurridos, capas, modo,
                cmap_al, norm_al, vmax, ms_ev=24, lw=1.1)
        axz.set_xlim(zb[0], zb[2])
        axz.set_ylim(zb[1], zb[3])
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_aspect("equal")
        axz.set_facecolor("white")
        for sp in axz.spines.values():
            sp.set_linewidth(1.1)
            sp.set_edgecolor("0.15")

        # Recuadro sobre el mapa grande y líneas hacia el zoom.
        _, lineas = ax.indicate_inset_zoom(
            axz, edgecolor="0.15", linewidth=1.0, alpha=1.0)
        # Se dejan solo las dos conexiones del lado indicado; las otras
        # cruzarían el mapa. Orden: (inf-izq, sup-izq, inf-der, sup-der).
        for i, ln in enumerate(lineas):
            ln.set_visible(i in INSET_CONECTORES)
            ln.set_linewidth(0.9)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    if modo == "alerta":
        handles = [Line2D([], [], marker="s", ls="", ms=11, color=c, label=l)
                   for c, l in zip(COLORES_ALERTA, NOMBRES_ALERTA)]
        if (su.get("evaluable", pd.Series(True, index=su.index)) == False).any():
            handles.append(Line2D([], [], marker="s", ls="", ms=11,
                                  color=COLOR_NO_EVAL, label="Not evaluated"))
        handles.append(Line2D([], [], marker="o", ls="", ms=8,
                              markerfacecolor="black", markeredgecolor="white",
                              label="Reported landslide"))
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.005), ncol=len(handles),
                   frameon=False, fontsize=9.5)
    else:
        sm = plt.cm.ScalarMappable(cmap="YlGnBu",
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
        cax = fig.add_axes([0.30, 0.045, 0.40, 0.016])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("STR, 24-hour antecedent rainfall (mm)", fontsize=9)

    base = os.path.join(OUT_DIR, f"fig_event_{tag}_{modo}")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", base + ".pdf")


# ============================================================
# FIGURA 3 — fotogramas del GIF
# ============================================================

def render_frame(ts, su, nivel, ev, ref, capas, ax_map, ax_zoom, ax_rain,
                 ax_acum, zb):
    cmap_al = ListedColormap(COLORES_MAPA)
    norm_al = BoundaryNorm(LIMITES_MAPA, cmap_al.N)
    ocurridos = ev[ev["fecha_hora"] <= ts]

    # El mapa del valle va SIN marco (solo marcas y coordenadas abajo y a la
    # izquierda), porque el contorno del valle no llena el rectángulo y el
    # marco encerraría mucho vacío. El zoom sí lo conserva.
    for ax, bounds, msize, coords, marco in ((ax_map, None, 14, True, False),
                                             (ax_zoom, zb, 34, True, True)):
        ax.clear()
        su.assign(v=nivel.to_numpy()).plot(
            ax=ax, column="v", cmap=cmap_al, norm=norm_al, linewidth=0)
        capas["rio"].plot(ax=ax, color=COLOR_RIO, linewidth=1.3, zorder=6)
        capas["municipios"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.9,
                                 zorder=7)
        if ax is ax_zoom:
            capas["comunas"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.6,
                                  zorder=7)
        if len(ocurridos):
            ocurridos.plot(ax=ax, color="black", edgecolor="white",
                           markersize=msize, linewidth=0.5, zorder=10)
        estilo_mapa(ax, bounds, coords=coords, marco=marco)

    ax_map.add_patch(Rectangle((zb[0], zb[1]), zb[2] - zb[0], zb[3] - zb[1],
                               fill=False, edgecolor="black", lw=1.2, zorder=11))
    ax_map.set_title("Aburrá Valley", fontsize=11)
    ax_zoom.set_title(f"Affected area · {ref['zona']}", fontsize=11)

    # --- hietograma + acumulado ---------------------------------------
    ax_rain.clear(); ax_acum.clear()
    ax_acum.yaxis.set_label_position("right"); ax_acum.yaxis.tick_right()
    idx, ih, ac = ref["idx"], ref["int_h"], ref["acum"]
    pasado = idx <= ts
    w = FRAME_HOURS / 24 * 0.9
    ax_rain.bar(idx[pasado], ih[pasado], width=w, color=COLOR_INT, zorder=3)
    ax_rain.bar(idx[~pasado], ih[~pasado], width=w, color="#B0BEC5",
                alpha=0.45, zorder=2)
    ax_acum.plot(idx[pasado], ac[pasado], color=COLOR_ACUM, lw=2, zorder=4)
    ax_acum.plot(idx, ac, color=COLOR_ACUM, lw=0.8, ls=":", alpha=0.5, zorder=3)
    ax_rain.axvline(ts, color="#C62828", lw=1.6, zorder=5)

    t_ev = ref["t_ev"]
    if t_ev is not None and not pd.isna(t_ev) and idx[0] <= t_ev <= idx[-1]:
        ax_rain.axvline(t_ev, color="black", lw=1, ls="--", zorder=4)
        ax_rain.annotate("landslides", xy=(t_ev, 0.97),
                         xycoords=("data", "axes fraction"), ha="right",
                         va="top", fontsize=7.5, rotation=90, clip_on=True)

    ax_rain.set_xlim(idx[0], idx[-1])
    ax_rain.set_ylim(0, max(float(np.nanmax(ih)), 1.0) * 1.25)
    ax_acum.set_ylim(0, max(float(np.nanmax(ac)), 1.0) * 1.15)
    ax_rain.set_ylabel("Intensity (mm/h)", color=COLOR_INT, fontsize=9)
    ax_acum.set_ylabel("Accumulated (mm)", color=COLOR_ACUM, fontsize=9)
    ax_rain.tick_params(axis="y", labelcolor=COLOR_INT, labelsize=8)
    ax_acum.tick_params(axis="y", labelcolor=COLOR_ACUM, labelsize=8)
    ax_rain.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax_rain.tick_params(axis="x", labelsize=8)
    ax_rain.set_title(f"Rain gauge {ref['cod']} "
                      f"({ref['dist_km']:.1f} km away)", fontsize=9)
    ax_rain.grid(alpha=0.25, zorder=0)


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = EVENTOS[EVENTO_ACTIVO]
    tag = EVENTO_ACTIVO
    os.makedirs(OUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUT_DIR, f"frames_{tag}")
    os.makedirs(frames_dir, exist_ok=True)

    su, gam, p_cortes, cache, oni, ev, gpl, capas = preparar(cfg)

    # --- zoom y pluviómetro de referencia -----------------------------
    # El margen y la distancia se calculan en metros y luego se pasan a
    # grados, no al revés: un grado de longitud mide menos que uno de
    # latitud, y a 6 N la diferencia ya es del 1 %.
    ev_m  = ev.to_crs(CRS_METRICO)
    gpl_m = gpl.to_crs(CRS_METRICO)
    x0m, y0m, x1m, y1m = ev_m.total_bounds
    zb_m = gpd.GeoSeries.from_xy(
        [x0m - ZOOM_MARGEN_M, x1m + ZOOM_MARGEN_M],
        [y0m - ZOOM_MARGEN_M, y1m + ZOOM_MARGEN_M],
        crs=CRS_METRICO).to_crs("EPSG:4326")
    zb = (zb_m.x.min(), zb_m.y.min(), zb_m.x.max(), zb_m.y.max())

    ev_union = ev_m.geometry.unary_union
    d_pl = gpl_m.geometry.distance(ev_union)      # metros
    cod_ref = int(gpl_m.loc[d_pl.idxmin(), "Codigo"])

    horas = pd.date_range(cfg["frame_start"], cfg["frame_end"],
                          freq=f"{FRAME_HOURS}H")
    lluvia_h = cache[cod_ref].diff().reindex(horas).fillna(0.0)
    ref = {"cod": cod_ref, "dist_km": float(d_pl.min()) / 1000.0,
           "idx": horas, "int_h": lluvia_h.to_numpy(float),
           "acum": lluvia_h.cumsum().to_numpy(float),
           "t_ev": ev["fecha_hora"].min(), "zona": cfg["nombre"]}
    print(f"Pluviómetro de referencia: {cod_ref} "
          f"a {ref['dist_km']:.1f} km del evento")

    # --- diagnóstico: vecino más cercano frente a campo interpolado ---
    # Los mapas usan el campo interpolado, pero el modelo está calibrado
    # con el vecino más cercano. Estos números son los que permiten
    # escribir en la tesis que la sustitución no altera la lectura.
    if METODO_LLUVIA == "kriging":
        t_pico = ev["fecha_hora"].dt.floor("H").mode()
        t_pico = pd.Timestamp(t_pico.iloc[0]) if len(t_pico) else horas[len(horas) // 2]
        d_cmp = comparar_metodos(t_pico, su, cache, p_cortes)
        d_cmp.to_csv(os.path.join(OUT_DIR, f"comparacion_lluvia_{tag}.csv"),
                     index=False)

    # --- figuras 1 y 2: rejillas de paneles ---------------------------
    if HORAS_PANEL is not None:
        hp = [pd.Timestamp(h) for h in HORAS_PANEL]
    else:
        hp = list(pd.to_datetime(
            np.linspace(horas[0].value, horas[-1].value, N_PANELES)).floor("H"))
    print("Horas de los paneles:", [f"{h:%d %b %H:%M}" for h in hp])

    rejilla_paneles(hp, su, gam, p_cortes, cache, oni, ev, capas, zb,
                    "alerta", tag, cfg["nombre"])
    rejilla_paneles(hp, su, gam, p_cortes, cache, oni, ev, capas, zb,
                    "lluvia", tag, cfg["nombre"])

    # --- figura 3: fotogramas del GIF ---------------------------------
    fig = plt.figure(figsize=(13.5, 7.6), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[2.2, 1],
                          hspace=0.30, wspace=0.18)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_rain = fig.add_subplot(gs[1, 1])
    ax_acum = ax_rain.twinx()

    leyenda = [Line2D([0], [0], marker="s", ls="", markersize=11, color=c,
                      label=n) for c, n in zip(COLORES_ALERTA, NOMBRES_ALERTA)]
    leyenda.append(Line2D([0], [0], marker="o", ls="", markersize=9,
                          markerfacecolor="black", markeredgecolor="white",
                          label="Reported landslide"))
    fig.legend(handles=leyenda, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=5, frameon=False, fontsize=9.5, title="Alert level",
               title_fontsize=10)

    frames, resumen = [], []
    for i, ts in enumerate(horas):
        nivel, R1, R30 = evaluar_valle(ts, su, gam, p_cortes, cache, oni)
        render_frame(ts, su, nivel, ev, ref, capas, ax_map, ax_zoom, ax_rain,
                     ax_acum, zb)
        fig.suptitle(f"{cfg['nombre']} · {ts:%d %B %Y %H:%M}",
                     fontsize=15, fontweight="bold")
        fpath = os.path.join(frames_dir, f"frame_{i:03d}.png")
        fig.savefig(fpath, bbox_inches="tight")
        frames.append(fpath)

        # resumen por hora: cuánto del valle está en cada nivel
        cuenta = pd.Series(nivel).value_counts().reindex([0, 1, 2, 3],
                                                         fill_value=0)
        resumen.append({"ts": ts, **{NOMBRES_ALERTA[k]: int(cuenta[k])
                                     for k in range(4)},
                        "pct_high_or_more": round(
                            100 * (cuenta[2] + cuenta[3]) / len(su), 2)})
        print(f"[{i+1}/{len(horas)}] {ts:%d %b %H:%M} | "
              f"very high: {int(cuenta[3])} SU")

    plt.close(fig)

    res = pd.DataFrame(resumen)
    res.to_csv(os.path.join(OUT_DIR, f"resumen_{tag}.csv"), index=False)
    print("\nResumen por hora guardado. Fracción del valle en High o "
          "Very high, máximo del evento: "
          f"{res['pct_high_or_more'].max():.1f}%")

    # --- GIF ----------------------------------------------------------
    import imageio.v3 as iio
    imgs = [iio.imread(f) for f in frames]
    gif = os.path.join(OUT_DIR, f"{tag}.gif")
    iio.imwrite(gif, imgs, duration=SEG_POR_FRAME * 1000, loop=0)
    print("GIF:", gif)


if __name__ == "__main__":
    main()
