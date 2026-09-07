"""
animacion_altavista_2025.py
===========================

Motor operacional + animación del evento del 29 de abril de 2025 (Altavista).

IDEA CENTRAL
------------
La función `evaluar_valle(ts)` ES el modelo operacional: dado un instante,
devuelve el nivel de alerta de cada Slope Unit. La animación solo la ejecuta
en bucle sobre el pasado. Lo que muestras en el simposio es la herramienta real.

FLUJO POR HORA
--------------
    1. R1 (24 h) y R30 (30 d) del pluviómetro más cercano CON DATOS a cada SU.
       Solo se usan pluviómetros cuya serie cubre toda la ventana de la
       animación (30 d antes del primer frame -> último frame); así ninguna
       SU válida queda sin pintar (parches blancos del primer borrador).
    2. R1 < 5 mm o R30 < 50 mm  ->  nivel de lluvia 0 (fuera de exposición).
    3. X = [T=R1, P=R30-R1, ENSO, zona, (tipo_su)] -> gam.predict_proba -> p
    4. p vs cortes P* (package.json)  ->  nivel de lluvia 0..3
    5. MATRIZ[clase_tasa][nivel]      ->  alerta 0..3 (Verde..Rojo)

FIGURA
------
    - Izquierda: Valle de Aburrá completo, con recuadro del zoom, contornos
      de municipios y río Aburrá.
    - Derecha arriba: zoom a la zona afectada (bbox de los deslizamientos),
      con marco negro y contornos de las comunas de Medellín.
    - Derecha abajo: hietograma (mm/h) + acumulado del evento (mm) del
      pluviómetro más cercano a los deslizamientos.
    - Deslizamientos observados: puntos NEGROS (el rojo es nivel de alerta).
    - Leyenda de alerta: Baja (verde), Media (amarillo), Alta (naranja),
      Muy Alta (rojo).

REQUIERE (salidas que ya tienes)
--------------------------------
    - su_resultados_ZINB_BYM2.gpkg  (clase_tasa, zona, tipo_su, incluir_parte2)
    - gam_*_run_star.joblib  +  gam_*_package.json   (del entrenamiento pooled)
    - pluviometros_metadatos_*.csv  +  series horarias H_Datos_Procesados_Est_*.csv
    - oni_diario_2010_2025.csv
    - CSV de deslizamientos del evento: fecha_hora, lon, lat  (DAGRD/SIATA)

USO
---
    python animacion_altavista_2025.py
"""

from __future__ import annotations   # compatibilidad de anotaciones con Python 3.8

import os
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ============================================================
# CONFIG — AJUSTAR RUTAS
# ============================================================

BASE          = "/home/oisanchezp/Thesis"
PATH_SU       = f"{BASE}/data/processed/su_resultados_ZINB_BYM2.gpkg"
LAYER_SU      = "clases_tasa"    # capa escrita por clases_tasa_esperada.R
PLUV_META     = f"{BASE}/data/metadata/pluviometros_metadatos_20250506.csv"
RUTA_SERIES   = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"
ONI_PATH      = f"{BASE}/data/metadata/oni_diario_2010_2025.csv"
GAM_MODEL     = f"{BASE}/src/06_Chapter/Umbrales/gam_pooled_run_star.joblib"
GAM_PACKAGE   = f"{BASE}/src/06_Chapter/Umbrales/gam_pooled_package.json"
EVENTOS_CSV   = f"{BASE}/data/altavista/eventos_altavista_20250428.csv"
#EVENTOS_CSV   = f"{BASE}/data/altavista/eventos_olivares_20250505.csv"
MUNI_SHP      = f"{BASE}/data/metadata/Valle_Aburra_Def_con_id.shp"   # veredas AMVA
RIO_GPKG      = f"{BASE}/data/metadata/rio_aburra.gpkg"
OUT_DIR       = f"{BASE}/src/06_Chapter/Umbrales/Animacion/"

# Columnas del GPKG de la Parte 1
COL_CLASE  = "clase_tasa"        # de clases_tasa_esperada.R (curva de éxito)
                                 # "Baja" | "Media" | "Alta" | "Muy alta"
                                 # NO usar clase_susc (mapa invertido, ver plan §1.1)
COL_ZONA   = "zona"              # debe coincidir con mappings["zona"] del package
COL_TIPO   = "tipo_su"           # solo si el GAM pooled usa su_type como factor
COL_INCLUIR = "incluir_parte2"   # ceros estructurales -> False (grises)

NOMBRE_ZONA_AFECTADA = "Altavista - Medellín"   # para el título del zoom

# Ventana de la animación (hora local)
FRAME_START = pd.Timestamp("2025-04-27 18:00")
FRAME_END   = pd.Timestamp("2025-04-28 04:00")
FRAME_HOURS = 1          # 1 frame por hora
SEG_POR_FRAME = 0.30     # duración de cada frame en el GIF

# Cortes P* — se leen del package.json
P_BAJO_KEY, P_MEDIO_KEY, P_ALTO_KEY = "P_TPR95", "P_TPR85", "P_TPR70"

# Exposición (idéntica al entrenamiento)
UMBRAL_1D, UMBRAL_30D = 5.0, 50.0

# Matriz de decisión: clase_tasa -> alerta según nivel de lluvia [0,1,2,3]
# Claves en minúscula: la clase se normaliza con .lower().
MATRIZ = {
    "muy alta": [0, 1, 2, 3],   # Verde, Amarillo, Naranja, Rojo
    "alta":     [0, 1, 1, 3],   # Verde, Amarillo, Amarillo, Rojo
    "media":    [0, 0, 1, 2],   # Verde, Verde,    Amarillo, Naranja
    "baja":     [0, 0, 0, 1],   # Verde, Verde,    Verde,    Amarillo
}
COLORES_ALERTA = ["#2E7D32", "#F9A825", "#EF6C00", "#C62828"]   # V, A, N, R
COLOR_EXCLUIDA = "#CFD8DC"                                       # cero estructural
# Etiquetas de los niveles de alerta en la leyenda
NOMBRES_ALERTA = ["Baja", "Media", "Alta", "Muy Alta"]
COLOR_INT      = "#0277BD"       # barras de intensidad
COLOR_ACUM     = "#263238"       # curva de acumulado
COLOR_LINDE    = "#37474F"       # contornos de municipios y comunas
COLOR_RIO      = "#0D47A1"       # río Aburrá (azul oscuro)

# Zoom a la zona afectada.
# None = automático: bbox de los deslizamientos + ZOOM_MARGEN_M de margen.
# También puedes fijarlo a mano: (x_min, y_min, x_max, y_max) en EPSG:32618.
ZOOM_BOUNDS   = None
ZOOM_MARGEN_M = 1500

# Posición de la leyenda de alerta (fracción de la FIGURA, x-y del centro).
# Vertical, sobre el hueco medio-inferior derecho del mapa del valle.
LEYENDA_POS   = (0.47, 0.30)


# ============================================================
# LLUVIA — mismas utilidades de tu script de umbrales
# ============================================================

def load_hourly_cumsum(cod: int) -> Optional[pd.Series]:
    path = os.path.join(RUTA_SERIES, f"H_Datos_Procesados_Est_{cod}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["Fecha", "P"], parse_dates=["Fecha"])
    serie = df.groupby("Fecha")["P"].sum().sort_index().asfreq("H", fill_value=0.0)
    return serie.cumsum().astype(np.float32)


def acumulado(cs: pd.Series, ts: pd.Timestamp, horas: int) -> float:
    """Lluvia acumulada en las últimas `horas` previas a ts."""
    if cs is None or cs.empty:
        return np.nan
    t = pd.Timestamp(ts).floor("H")
    # fuera del rango de la serie -> NaN (antes 'pad' devolvía lluvia vieja)
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
# PREPARACIÓN (una sola vez)
# ============================================================

def cargar_capas_contexto(crs):
    """Contornos de municipios, comunas de Medellín y río Aburrá (estética)."""
    ver = gpd.read_file(MUNI_SHP).to_crs(crs)
    ver["geometry"] = ver.geometry.buffer(0)           # sanea topología p/ dissolve
    municipios = ver.dissolve("Municipio").boundary
    es_med  = ver["Municipio"].astype(str).str.upper().str.contains("MEDELL")
    comunas = ver[es_med].dissolve("Comuna").boundary  # incluye corr. Altavista
    rio = gpd.read_file(RIO_GPKG).to_crs(crs)

    # Punto de la etiqueta "Río Aburrá": vértice más al norte (donde el río
    # sale del valle) + ángulo local del cauce para rotar el texto.
    lineas = []
    for g in rio.geometry:
        if g.geom_type == "MultiLineString":
            lineas.extend(g.geoms)
        elif g.geom_type == "LineString":
            lineas.append(g)
    rio_label = None
    for ln in lineas:
        c = np.asarray(ln.coords)[:, :2]
        i = int(np.argmax(c[:, 1]))
        if rio_label is None or c[i, 1] > rio_label[1]:
            j = i - 1 if i > 0 else i + 1
            ang = float(np.degrees(np.arctan2(c[i, 1] - c[j, 1],
                                              c[i, 0] - c[j, 0])))
            if ang > 90:   ang -= 180          # texto nunca al revés
            elif ang < -90: ang += 180
            rio_label = (float(c[i, 0]), float(c[i, 1]), ang)

    print(f"Contexto: {len(municipios)} municipios, {len(comunas)} comunas de "
          f"Medellín, río Aburrá ({len(rio)} geom.)")
    return {"municipios": municipios, "comunas": comunas, "rio": rio,
            "rio_label": rio_label}


def preparar():
    # --- Capa de SU con resultados de la Parte 1 ---
    su = gpd.read_file(PATH_SU, layer=LAYER_SU).to_crs("EPSG:32618")
    su["geometry"] = su.geometry.simplify(15)          # acelera el render x frame

    # --- GAM + package ---
    gam = joblib.load(GAM_MODEL)
    with open(GAM_PACKAGE, encoding="utf-8") as f:
        pkg = json.load(f)
    mappings  = pkg["mappings"]
    features  = pkg["feature_names"]                   # 4 o 5 features
    thr       = pkg["summary_mc"]["thresholds"]
    p_cortes  = (thr[P_BAJO_KEY]["median"],
                 thr[P_MEDIO_KEY]["median"],
                 thr[P_ALTO_KEY]["median"])
    print(f"P* bajo/medio/alto = {p_cortes}")

    # --- Codificar factores de cada SU (fijos en el tiempo) ---
    def _norm_key(serie):
        """'2', '2.0', 2.0 -> '2' para casar con las claves del package."""
        v = serie.astype(str).str.strip()
        try:
            return v.astype(float).astype(int).astype(str)
        except ValueError:
            return v
    su["zona_code"] = _norm_key(su[COL_ZONA]).map(mappings["zona"])
    if "su_type" in mappings:                          # GAM pooled con tipo de SU
        su["tipo_code"] = _norm_key(su[COL_TIPO]).map(mappings["su_type"])
    faltan = su["zona_code"].isna().sum()
    if faltan:
        raise ValueError(
            f"{faltan} SU con zona fuera de mappings['zona']={list(mappings['zona'])}. "
            f"Valores en la capa: {sorted(su[COL_ZONA].astype(str).unique())}"
        )

    # --- Pluviómetros: SOLO los que cubren toda la ventana de la animación ---
    # CAUSA de los parches blancos del primer borrador: se asignaba el
    # pluviómetro más cercano aunque no tuviera serie utilizable, y luego esas
    # SU se eliminaban del GeoDataFrame (no se pintaban). Ahora se filtran
    # ANTES del sjoin: cada SU recibe el más cercano CON datos completos.
    pl = pd.read_csv(PLUV_META)
    gpl = gpd.GeoDataFrame(
        pl, geometry=gpd.points_from_xy(pl["Longitude"], pl["Latitude"]),
        crs="EPSG:4326").to_crs(su.crs)
    gpl["Codigo"] = gpl["Codigo"].astype(int)

    ini_req = FRAME_START - pd.Timedelta(days=31)      # R30 necesita 30 d antes
    cache = {}
    for cod in gpl["Codigo"].unique():
        cs = load_hourly_cumsum(int(cod))
        if cs is None or cs.empty:
            continue
        if cs.index[0] <= ini_req and cs.index[-1] >= FRAME_END:
            cache[int(cod)] = cs
    gpl = gpl[gpl["Codigo"].isin(cache)].copy()
    print(f"Pluviómetros con serie completa {ini_req:%Y-%m-%d} -> "
          f"{FRAME_END:%Y-%m-%d}: {len(gpl)}")
    if gpl.empty:
        raise RuntimeError("Ningún pluviómetro cubre la ventana de la animación")

    # --- Pluviómetro (con datos) más cercano por SU ---
    try:
        su = gpd.sjoin_nearest(su, gpl[["Codigo", "geometry"]],
                               how="left", distance_col="dist_pluv_m")
        # Si su ya traía una columna 'Codigo' propia, geopandas sufija la unión
        # como 'Codigo_right' en vez de 'Codigo'. Cubrimos ambos casos.
        if "Codigo_right" in su.columns:
            su = su.rename(columns={"Codigo_right": "Codigo_pluvio",
                                    "Codigo_left": "Codigo"})
        else:
            su = su.rename(columns={"Codigo": "Codigo_pluvio"})
        su = su.drop(columns=[c for c in ("index_right",) if c in su.columns])
        # sjoin_nearest duplica filas si hay empate de distancia: nos quedamos con una
        su = su[~su.index.duplicated(keep="first")]
    except Exception:
        # Fallback para geopandas viejos (sin sjoin_nearest): cKDTree de scipy
        from scipy.spatial import cKDTree
        cent = su.geometry.centroid
        arbol = cKDTree(np.column_stack([gpl.geometry.x, gpl.geometry.y]))
        dist, idx = arbol.query(np.column_stack([cent.x, cent.y]), k=1)
        su["Codigo_pluvio"] = gpl["Codigo"].to_numpy()[idx]
        su["dist_pluv_m"] = dist
    if "Codigo_pluvio" not in su.columns:
        raise RuntimeError("No se pudo asignar pluviómetro: revisa columnas de su y gpl")
    sin = int(su["Codigo_pluvio"].isna().sum())
    if sin:
        raise RuntimeError(f"{sin} SU quedaron sin pluviómetro asignado")
    su["Codigo_pluvio"] = su["Codigo_pluvio"].astype(int)
    print(f"SU: {len(su)} (ninguna descartada; dist. máx. a pluvio: "
          f"{su['dist_pluv_m'].max():.0f} m)")

    # --- ONI diario -> fase ENSO ---
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    oni["ENSO"] = oni["ENSO"].astype(str).str.strip().replace({"Neutral": "Neutro"})
    oni = oni.set_index(oni["date"].dt.normalize())["ENSO"]
    oni = oni[~oni.index.duplicated()]

    # --- Eventos observados ---
    ev = pd.read_csv(EVENTOS_CSV, parse_dates=["fecha_hora"])
    ev = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["lon"], ev["lat"]),
                          crs="EPSG:4326").to_crs(su.crs)

    # --- Capas de contexto (municipios, comunas, río) ---
    capas = cargar_capas_contexto(su.crs)

    return su, gam, mappings, features, p_cortes, cache, oni, ev, gpl, capas


# ============================================================
# MOTOR OPERACIONAL — la misma función que correría cada hora
# ============================================================

def evaluar_valle(ts, su, gam, mappings, features, p_cortes, cache, oni):
    """Devuelve la serie de alerta (0..3, -1=excluida) por SU en el instante ts."""
    ts = pd.Timestamp(ts)

    # 1. lluvia por pluviómetro (no por SU: pocos cientos vs ~50k)
    r1, r30 = {}, {}
    for cod, cs in cache.items():
        r1[cod]  = acumulado(cs, ts, 24)
        r30[cod] = acumulado(cs, ts, 30 * 24)
    su = su.copy()
    su["R1"]  = su["Codigo_pluvio"].map(r1)
    su["R30"] = su["Codigo_pluvio"].map(r30)

    # 2. fase ENSO del día
    enso = oni.get(ts.normalize(), None)
    enso_code = mappings["ENSO"].get(str(enso), None)
    if enso_code is None:
        raise ValueError(f"Fase ENSO desconocida para {ts}: {enso}")

    # 3. probabilidad solo donde hay exposición y la SU no es cero estructural
    activa = (su["R1"] >= UMBRAL_1D) & (su["R30"] >= UMBRAL_30D) \
             & su[COL_INCLUIR].astype(bool) & su["R1"].notna() & su["R30"].notna()

    nivel = np.zeros(len(su), dtype=int)
    if activa.any():
        d = su.loc[activa]
        cols = {
            "T":         d["R1"].to_numpy(float),
            "P":         (d["R30"] - d["R1"]).clip(lower=0).to_numpy(float),
            "ENSO_code": np.full(activa.sum(), enso_code, float),
            "zona_code": d["zona_code"].to_numpy(float),
        }
        # Si el GAM pooled incluye el tipo de SU, el package debe nombrar
        # la feature exactamente "su_type_code" (5.ª posición).
        if len(features) == 5:
            cols["su_type_code"] = d["tipo_code"].to_numpy(float)
        X = np.column_stack([cols[fn] for fn in features])
        p = gam.predict_proba(X)
        n = np.zeros(len(p), dtype=int)
        n[p >= p_cortes[0]] = 1
        n[p >= p_cortes[1]] = 2
        n[p >= p_cortes[2]] = 3
        nivel[activa.to_numpy()] = n

    # 4. matriz de decisión (clase normalizada a minúsculas)
    clases = su[COL_CLASE].astype(str).str.strip().str.lower()
    alerta = np.array([MATRIZ[c][k] for c, k in zip(clases, nivel)])
    alerta[~su[COL_INCLUIR].astype(bool).to_numpy()] = -1        # gris
    return pd.Series(alerta, index=su.index), su["R1"], su["R30"]


# ============================================================
# ANIMACIÓN
# ============================================================

def render_frame(ts, su, alerta, ev, ref, capas, ax_map, ax_zoom, ax_rain,
                 ax_acum, zb):
    colores = pd.Series(COLOR_EXCLUIDA, index=su.index)
    for k in range(4):
        colores[alerta == k] = COLORES_ALERTA[k]

    # --- Mapas: valle completo + zoom a la zona afectada ---
    ocurridos = ev[ev["fecha_hora"] <= ts]
    for ax, bounds, msize in ((ax_map, None, 14), (ax_zoom, zb, 32)):
        ax.clear()
        su.plot(ax=ax, color=colores, linewidth=0)
        # contexto: río Aburrá + contornos administrativos
        capas["rio"].plot(ax=ax, color=COLOR_RIO, linewidth=1.3, zorder=6)
        capas["municipios"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.9,
                                 zorder=7)
        if ax is ax_zoom:      # comunas de Medellín solo en el zoom
            capas["comunas"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.6,
                                  zorder=7)
        if len(ocurridos):
            # NEGRO con borde blanco: el rojo queda reservado al nivel de alerta
            ocurridos.plot(ax=ax, color="black", edgecolor="white",
                           markersize=msize, linewidth=0.5, zorder=10)
        if bounds is not None:
            ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
    # recuadro de ubicación del zoom sobre el mapa general
    ax_map.add_patch(Rectangle((zb[0], zb[1]), zb[2] - zb[0], zb[3] - zb[1],
                               fill=False, edgecolor="black", lw=1.1, zorder=11))
    # etiqueta del río en su salida del valle, paralela al cauce
    if capas.get("rio_label"):
        xl, yl, angl = capas["rio_label"]
        ax_map.annotate("Río Aburrá", xy=(xl, yl), xytext=(-5, 4),
                        textcoords="offset points", color=COLOR_RIO,
                        fontsize=8.5, fontstyle="italic", ha="right",
                        va="bottom", rotation=angl, rotation_mode="anchor",
                        zorder=8)
    ax_map.set_axis_off()
    # el zoom conserva su marco NEGRO (sin ticks)
    ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
    for sp in ax_zoom.spines.values():
        sp.set_visible(True); sp.set_edgecolor("black"); sp.set_linewidth(1.3)
    ax_map.set_title("Valle de Aburrá", fontsize=11)
    ax_zoom.set_title(f"Zona afectada · {NOMBRE_ZONA_AFECTADA}", fontsize=11)

    # --- Hietograma + acumulado del pluviómetro más cercano al evento ---
    ax_rain.clear(); ax_acum.clear()
    # clear() resetea el twinx: devolver etiqueta y ticks al lado DERECHO
    ax_acum.yaxis.set_label_position("right")
    ax_acum.yaxis.tick_right()
    idx, ih, ac = ref["idx"], ref["int_h"], ref["acum"]
    pasado = idx <= ts
    w = FRAME_HOURS / 24 * 0.9                     # ancho de barra en días
    ax_rain.bar(idx[pasado], ih[pasado], width=w, color=COLOR_INT, zorder=3)
    ax_rain.bar(idx[~pasado], ih[~pasado], width=w, color="#B0BEC5",
                alpha=0.45, zorder=2)              # futuro atenuado
    ax_acum.plot(idx[pasado], ac[pasado], color=COLOR_ACUM, lw=2, zorder=4)
    ax_acum.plot(idx, ac, color=COLOR_ACUM, lw=0.8, ls=":", alpha=0.5, zorder=3)
    ax_rain.axvline(ts, color="#C62828", lw=1.6, zorder=5)       # "ahora"
    # marca de la hora de los deslizamientos SOLO si cae dentro de la ventana
    # (fuera de ella el texto quedaba flotando a la derecha del panel)
    t_ev = ref["t_ev"]
    if t_ev is not None and not pd.isna(t_ev) and idx[0] <= t_ev <= idx[-1]:
        ax_rain.axvline(t_ev, color="black", lw=1, ls="--", zorder=4)
        ax_rain.annotate("deslizamientos", xy=(t_ev, 0.97),
                         xycoords=("data", "axes fraction"),
                         ha="right", va="top", fontsize=7.5, rotation=90,
                         clip_on=True)

    ax_rain.set_xlim(idx[0], idx[-1])
    ax_rain.set_ylim(0, max(float(np.nanmax(ih)), 1.0) * 1.25)
    ax_acum.set_ylim(0, max(float(np.nanmax(ac)), 1.0) * 1.15)
    ax_rain.set_ylabel("Intensidad (mm/h)", color=COLOR_INT, fontsize=9)
    ax_acum.set_ylabel("Acumulado (mm)", color=COLOR_ACUM, fontsize=9)
    ax_rain.tick_params(axis="y", labelcolor=COLOR_INT, labelsize=8)
    ax_acum.tick_params(axis="y", labelcolor=COLOR_ACUM, labelsize=8)
    ax_rain.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    ax_rain.tick_params(axis="x", labelsize=8)
    ax_rain.set_title(f"Pluviómetro {ref['cod']}", fontsize=9)
    ax_rain.grid(alpha=0.25, zorder=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    su, gam, mappings, features, p_cortes, cache, oni, ev, gpl, capas = preparar()

    # Fechas en español para el reloj del título (si el locale existe)
    import locale
    for loc in ("es_CO.UTF-8", "es_ES.UTF-8", "es_CO", "es_ES"):
        try:
            locale.setlocale(locale.LC_TIME, loc); break
        except locale.Error:
            continue

    # --- Zoom: bbox de los deslizamientos + margen (o el fijado a mano) ---
    if ZOOM_BOUNDS is not None:
        zb = ZOOM_BOUNDS
    else:
        x0, y0, x1, y1 = ev.total_bounds
        zb = (x0 - ZOOM_MARGEN_M, y0 - ZOOM_MARGEN_M,
              x1 + ZOOM_MARGEN_M, y1 + ZOOM_MARGEN_M)

    # --- Pluviómetro de referencia: el más cercano a los deslizamientos ---
    ev_union = ev.geometry.unary_union
    d_pl = gpl.geometry.distance(ev_union)
    cod_ref = int(gpl.loc[d_pl.idxmin(), "Codigo"])
    horas = pd.date_range(FRAME_START, FRAME_END, freq=f"{FRAME_HOURS}H")
    cs_ref = cache[cod_ref]
    lluvia_h = cs_ref.diff().reindex(horas).fillna(0.0)          # mm por hora
    ref = {
        "cod":      cod_ref,
        "dist_km":  float(d_pl.min()) / 1000.0,
        "idx":      horas,
        "int_h":    lluvia_h.to_numpy(float),
        "acum":     lluvia_h.cumsum().to_numpy(float),           # acumulado del evento
        "r30_ini":  acumulado(cs_ref, FRAME_START, 30 * 24),     # antecedente
        "t_ev":     ev["fecha_hora"].min(),
    }
    print(f"Pluviómetro de referencia: {cod_ref} a {ref['dist_km']:.1f} km")

    fig = plt.figure(figsize=(13.5, 7.6), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[2.2, 1],
                          hspace=0.30, wspace=0.05)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_rain = fig.add_subplot(gs[1, 1])
    ax_acum = ax_rain.twinx()          # creado UNA vez (twinx en el bucle los acumularía)

    leyenda = [Line2D([0], [0], marker="s", ls="", markersize=11, color=c,
                      label=n) for c, n in zip(COLORES_ALERTA, NOMBRES_ALERTA)]
    leyenda.append(Line2D([0], [0], marker="o", ls="", markersize=9,
                          markerfacecolor="black", markeredgecolor="white",
                          label="Deslizamiento observado"))
    # La leyenda se crea UNA vez (fig.legend en el bucle las acumularía) y es
    # de FIGURA, no del eje: ax_map.clear() la borraría en cada frame.
    # Vertical y compacta, anclada dentro del mapa del valle (LEYENDA_POS).
    fig.legend(handles=leyenda, loc="center", bbox_to_anchor=LEYENDA_POS,
               ncol=1, frameon=False, fontsize=9, title="Nivel de alerta",
               title_fontsize=9, labelspacing=0.35, handletextpad=0.5,
               borderaxespad=0.0)

    frames = []
    for i, ts in enumerate(horas):
        alerta, _, _ = evaluar_valle(ts, su, gam, mappings, features,
                                     p_cortes, cache, oni)
        render_frame(ts, su, alerta, ev, ref, capas, ax_map, ax_zoom, ax_rain,
                     ax_acum, zb)
        fig.suptitle(f"{ts:%A %d de %B de %Y · %H:%M}", fontsize=15,
                     fontweight="bold")
        fpath = os.path.join(OUT_DIR, f"frame_{i:03d}.png")
        fig.savefig(fpath, bbox_inches="tight")
        frames.append(fpath)
        print(f"[{i+1}/{len(horas)}] {ts}  alertas rojas: {(alerta == 3).sum()}")

    # GIF + MP4
    import imageio.v3 as iio
    imgs = [iio.imread(f) for f in frames]
    iio.imwrite(os.path.join(OUT_DIR, "altavista_20250428.gif"), imgs,
                duration=SEG_POR_FRAME * 1000, loop=0)
    try:
        iio.imwrite(os.path.join(OUT_DIR, "altavista_20250428.mp4"), imgs,
                    fps=int(1 / SEG_POR_FRAME))
    except Exception as e:
        print(f"MP4 no disponible ({e}); usa el GIF o ffmpeg sobre los frames.")
    print("Animación lista en", OUT_DIR)


if __name__ == "__main__":
    main()