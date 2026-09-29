"""Mapas del capítulo 6 con lluvia interpolada en malla de 100 m.

Copia independiente de mapas_alerta_cap6.py. Acumula por estación con las
funciones originales, interpola componentes temporales no solapados usando
kriging esférico log1p y obtiene medias de área por SU. Conserva GAM/cortes/matriz.
Respeta LTR_DEF del modelo: R30-R1 son 29 días previos, R31-R1 son 30 días.
Los campos R30/RLTR se reconstruyen sumando componentes interpolados en mm.
Incluye comparación retrospectiva con la asignación al pluviómetro más cercano.
El hietograma corresponde a una estación de referencia, no a lluvia media de SU.
Ver LEEME.md para método, dependencias, archivos y límites de las comprobaciones.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.path import Path
from matplotlib.collections import PathCollection
from shapely.geometry import box
from shapely.geometry.polygon import orient


# ============================================================
# CONFIGURACIÓN
# ============================================================

def _env(nombre: str, defecto: str) -> str:
    return os.environ.get(nombre, defecto)


BASE    = _env("CAP6_BASE", "/home/oisanchezp/Thesis")
CH5_DIR = _env("CAP6_CH5_DIR", f"{BASE}/src/05_Chapter")        # mc_common.py
CH6_DIR = _env("CAP6_CH6_DIR", f"{BASE}/src/06_Chapter")

# Salidas de gam_umbrales_cap6.py (el diseño que se reporta en la tesis)
DISENO  = _env("CAP6_DISENO", "excl10d_libre")
UMB_DIR = _env("CAP6_UMB_DIR", f"{CH6_DIR}/Umbrales/{DISENO}")
TAG_GAM = "gam_cap6"

# Slope units con la clase de susceptibilidad y el dominio (salida del qmd)
PATH_SU  = _env("CAP6_PATH_SU", f"{CH6_DIR}/salidas/su_resultados_ZINB_BYM2.gpkg")
LAYER_SU = "susceptibilidad"

PLUV_META   = _env("CAP6_PLUV_META", f"{BASE}/data/metadata/pluviometros_metadatos_20250506.csv")
RUTA_SERIES = _env("CAP6_RUTA_SERIES",
                   "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/")
ONI_PATH    = _env("CAP6_ONI_PATH", f"{BASE}/data/metadata/oni_diario_2010_2025.csv")
MUNI_SHP    = _env("CAP6_MUNI_SHP", f"{BASE}/data/metadata/Valle_Aburra_Def_con_id.shp")
RIO_GPKG    = _env("CAP6_RIO_GPKG", f"{BASE}/data/metadata/rio_aburra.gpkg")
EVENTOS_DIR = _env("CAP6_EVENTOS_DIR", f"{BASE}/data/altavista")
OUT_DIR     = _env("CAP6_OUT_INTERPOLADO", f"{CH6_DIR}/Mapas_alerta_kriging100m/")

# --- eventos: mismas ventanas que en el Capítulo 5 --------------------
EVENTOS = {
    "altavista": dict(
        nombre      = "Altavista",
        csv         = f"{EVENTOS_DIR}/eventos_altavista_20250428.csv",
        frame_start = pd.Timestamp("2025-04-27 18:00"),
        frame_end   = pd.Timestamp("2025-04-28 04:00"),
    ),
    "olivares": dict(
        nombre      = "Olivares",
        csv         = f"{EVENTOS_DIR}/eventos_olivares_20250505.csv",
        frame_start = pd.Timestamp("2025-05-04 19:00"),
        frame_end   = pd.Timestamp("2025-05-05 07:00"),
    ),
    "granizal": dict(
        nombre      = "Granizal",
        csv         = f"{EVENTOS_DIR}/eventos_granizal_20250624.csv",
        frame_start = pd.Timestamp("2025-06-23 19:00"),
        frame_end   = pd.Timestamp("2025-06-24 06:00"),
    ),
}
EVENTOS_A_CORRER = ["altavista", "olivares", "granizal"]

# Figuras de rejilla: "alerta" (matriz), "clase_lluvia" (GAM solo) y
# "lluvia" (STR en mm; es la misma lluvia de los mapas del Cap. 5).
FIGURAS       = ["alerta", "clase_lluvia", "lluvia"]
HACER_FRAMES  = True
FRAME_HOURS   = 1        # un fotograma por hora
SEG_POR_FRAME = 0.30     # duración de cada fotograma en el GIF
ZOOM_MARGEN_M = 1500     # margen del zoom alrededor de los deslizamientos

# Recuadro del zoom dentro de cada panel [x0, y0, ancho, alto] (fracción de
# los ejes) y conexiones visibles (0 inf-izq, 1 sup-izq, 2 inf-der, 3 sup-der).
INSET_POS        = [0.56, 0.03, 0.42, 0.42]
INSET_CONECTORES = (0, 1)

HORAS_PANEL = None       # None = N_PANELES horas repartidas en la ventana
N_PANELES   = 6          # 4 o 6

# CRS del dibujo. Las distancias (pluviómetro, zoom) se calculan en el CRS
# métrico de las slope units (EPSG:32618), el mismo del entrenamiento.
CRS_DIBUJO = "EPSG:4326"

# --- columnas de la capa de slope units --------------------------------
COL_ID      = "su_id"
COL_CLASE   = "clase_susc"      # Baja | Media | Alta | Muy alta
COL_ZONA    = "zona"
COL_TIPO    = "tipo_su"
COL_INCLUIR = "incluir_parte2"  # dominio del componente de lluvia

# --- clases y matriz de decisión ----------------------------------------
# El corte MÁS PERMISIVO (TPR95) abre la clase de lluvia MÁS BAJA de alerta.
CLAVES_CORTE  = ["P_TPR95", "P_TPR85", "P_TPR70"]
CLASES_LLUVIA = ["Low", "Medium", "High", "Very high"]
CLASES_SUSC   = ["Baja", "Media", "Alta", "Muy alta"]      # como en el gpkg
SUSC_EN       = {"Baja": "Low", "Media": "Moderate", "Alta": "High", "Muy alta": "Very high"}

# MATRIZ[susceptibilidad][clase de lluvia Low, Medium, High, Very high] -> nivel
# Idéntica a LEVEL de fig_matriz_decision.py. Si decides subir la fila Alta
# con lluvia alta (pendiente en decisiones_cap6.md), cambia aquí [1, 2, 4, 4].
MATRIZ = {
    "Baja":     [1, 2, 2, 3],
    "Media":    [1, 2, 3, 4],
    "Alta":     [1, 2, 3, 4],
    "Muy alta": [1, 3, 4, 4],
}
NIVELES        = [1, 2, 3, 4]
NOMBRES_NIVEL  = {1: "I", 2: "II", 3: "III", 4: "IV"}
NIVEL_AVISO    = 3      # "warned" = nivel III o IV en alguna hora (Métodos)

# Niveles de alerta: los colores del Cap. 5 y de la matriz (allí, aclarados).
COLORES_NIVEL = {1: "#2E7D32", 2: "#F9A825", 3: "#EF6C00", 4: "#C62828"}
# Clases de lluvia: por defecto, los colores RdYlBu_r de las figuras de umbrales
# del capítulo (plano STR-LTR y ROC), para que el mapa de la clase de lluvia no
# se confunda con el de niveles de alerta. "cap5" repite los colores de alerta.
PALETA_CLASES = _env("CAP6_PALETA_CLASES", "RdYlBu_r")      # "RdYlBu_r" | "cap5"
COLORES_CLASE = ({0: "#4574B3", 1: "#FEE090", 2: "#F8864F", 3: "#CC2627"}
                 if PALETA_CLASES == "RdYlBu_r" else
                 {k - 1: c for k, c in COLORES_NIVEL.items()})
FUERA_DOMINIO, SIN_LLUVIA = -1, -2
COLOR_FUERA   = "#BDBDBD"       # fuera del dominio de análisis
COLOR_SIN     = "#EEEEEE"       # sin pluviómetro válido (no debería aparecer)

COLOR_LINDE = "#37474F"
COLOR_RIO   = "#0D47A1"
COLOR_INT   = "#0277BD"
COLOR_ACUM  = "#263238"


# ============================================================
# mc_common (lluvia EXACTAMENTE como en el entrenamiento)
# ============================================================

if not os.path.isfile(os.path.join(CH5_DIR, "mc_common.py")):
    raise FileNotFoundError(f"No encuentro mc_common.py en {CH5_DIR}. Ajusta CH5_DIR.")
if CH5_DIR not in sys.path:
    sys.path.insert(0, CH5_DIR)
import mc_common as mcc  # noqa: E402
from lluvia_interpolada import ZonalArea, interpolar_evento

mcc.CFG.update({"PLUV_META": PLUV_META, "RUTA_SERIES": RUTA_SERIES, "ONI_PATH": ONI_PATH})
logging.basicConfig(level=logging.WARNING, format="%(message)s")


# ============================================================
# MODELO: efectos aditivos del GAM final y cortes medianos
# ============================================================

def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def componentes(e: pd.DataFrame) -> dict:
    """Piezas aditivas del GAM a partir de su tabla de efectos (como el notebook)."""
    g = lambda t: e[e["term"] == t]
    return {"b0": float(g("intercept")["effect"].iloc[0]),
            "STR": (g("STR")["x"].to_numpy(float), g("STR")["effect"].to_numpy(float)),
            "LTR": (g("LTR")["x"].to_numpy(float), g("LTR")["effect"].to_numpy(float)),
            "ENSO": dict(zip(g("ENSO")["level"], g("ENSO")["effect"].astype(float))),
            "zona": dict(zip(g("zona")["level"], g("zona")["effect"].astype(float))),
            "tipo_su": dict(zip(g("tipo_su")["level"], g("tipo_su")["effect"].astype(float)))}


def cargar_modelo() -> Tuple[dict, List[float], dict]:
    """Efectos del modelo final, cortes medianos y configuración del entrenamiento."""
    P = lambda n: os.path.join(UMB_DIR, f"{TAG_GAM}_{n}")
    with open(P("resumen.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    cfg = res["config"]
    comp = componentes(pd.read_csv(P("efectos_final.csv"), dtype={"level": str}))

    thr = pd.read_csv(P("mc_thresholds.csv"))
    med = thr.groupby("thr_level")["threshold"].median()
    cortes = [float(med[k]) for k in CLAVES_CORTE]
    if not np.all(np.diff(cortes) > 0):
        raise ValueError(f"Los cortes deben crecer (P95 < P85 < P70): {cortes}")

    fin = res["modelo_final"]
    print(f"Modelo final: corrida {fin['run_star']} | diseño {cfg.get('DISENO_TEMPORAL', DISENO)} "
          f"| LTR = {cfg['LTR_DEF']} | filtro R1 > {cfg['UMBRAL_1D']:g} mm y "
          f"R30 > {cfg['UMBRAL_30D']:g} mm")
    print("Cortes p* (mediana de las corridas): " +
          " | ".join(f"{k} = {v:.3f}" for k, v in zip(CLAVES_CORTE, cortes)))
    return comp, cortes, cfg


# ============================================================
# SLOPE UNITS, PLUVIÓMETROS, ONI, EVENTOS
# ============================================================

def _nivel_txt(v) -> str:
    """'2', '2.0', 2 -> '2' (las claves de zona y tipo del GAM son texto)."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v).strip()


def cargar_su(comp: dict) -> gpd.GeoDataFrame:
    """Slope units con clase de susceptibilidad, dominio, zona y tipo."""
    su = gpd.read_file(PATH_SU, layer=LAYER_SU)
    faltan = [c for c in (COL_ID, COL_CLASE, COL_ZONA, COL_TIPO, COL_INCLUIR)
              if c not in su.columns]
    if faltan:
        raise KeyError(f"Faltan {faltan} en {PATH_SU}. Hay: {sorted(su.columns)}")
    if su.crs is None:
        raise ValueError("La capa de slope units no tiene CRS.")
    if su.crs.is_geographic:
        su = su.to_crs("EPSG:32618")
    if su[COL_ID].duplicated().any():
        raise ValueError(f"{COL_ID} repetido en la capa de slope units.")

    su = su.reset_index(drop=True)
    su["dominio"] = su[COL_INCLUIR].astype(bool)
    su["clase_susc_txt"] = su[COL_CLASE].astype(str).str.strip()
    raras = sorted(set(su.loc[su["dominio"], "clase_susc_txt"]) - set(MATRIZ))
    if raras:
        raise ValueError(f"Clases de susceptibilidad sin fila en MATRIZ: {raras}")
    su["susc_idx"] = su["clase_susc_txt"].map({c: i for i, c in enumerate(CLASES_SUSC)})

    # Efecto de zona y tipo de cada SU del dominio (fijos en el tiempo)
    for col, clave in ((COL_ZONA, "zona"), (COL_TIPO, "tipo_su")):
        txt = su[col].map(_nivel_txt)
        ef = txt.map(comp[clave])
        sin = su["dominio"] & ef.isna()
        if sin.any():
            raise ValueError(f"{int(sin.sum())} SU del dominio con {col} fuera del modelo: "
                             f"{sorted(txt[sin].unique())} (el GAM tiene {sorted(comp[clave])})")
        su[f"ef_{clave}"] = ef.astype(float)

    n_dom = int(su["dominio"].sum())
    print(f"Slope units: {len(su)} | en el dominio: {n_dom} "
          f"({100 * n_dom / len(su):.1f} %) | por clase (dominio): "
          + ", ".join(f"{c} {int(((su['clase_susc_txt'] == c) & su['dominio']).sum())}"
                      for c in CLASES_SUSC))
    return su


def cargar_series(codigos) -> Dict[int, tuple]:
    """(cs, cv) de cada pluviómetro, con la función del entrenamiento."""
    cache = {}
    for cod in sorted(set(int(c) for c in codigos)):
        cs, cv = mcc.load_gauge_series(cod)
        if cs is not None and len(cs):
            cache[cod] = (cs, cv)
    print(f"Pluviómetros con serie: {len(cache)}")
    return cache


def _lluvia(cs, cv, ts, k) -> float:
    """mcc.rain_at, pero NaN si la serie del pluviómetro no llega hasta ts.

    rain_at ubica ts con method="pad": si la serie terminó antes del evento,
    devolvería la ventana de su último día (lluvia vieja, no la del evento).
    En la tabla del entrenamiento no hay ningún registro en esa situación.
    """
    if pd.Timestamp(ts).floor("h") not in cs.index:
        return np.nan
    return mcc.rain_at(cs, cv, ts, k)


def lluvia_evento(horas, cache, cfg) -> Tuple[Dict[str, pd.DataFrame], List[int]]:
    """R1, R30 y la ventana larga del LTR (mm) por hora y pluviómetro.

    Devuelve las tablas (filas = horas, columnas = pluviómetros) y la lista de
    pluviómetros con las tres lluvias válidas en TODAS las horas del evento.
    """
    k_ltr = 31 if str(cfg.get("LTR_DEF", "R30-R1")).upper().startswith("R31") else 30
    ventanas = {"R1": 1, "R30": 30, "RLTR": k_ltr}
    tablas = {}
    for nombre, k in ventanas.items():
        tablas[nombre] = pd.DataFrame(
            {cod: [_lluvia(cs, cv, ts, k) for ts in horas] for cod, (cs, cv) in cache.items()},
            index=horas, dtype=float)
    validos = [c for c in cache if all(tablas[n][c].notna().all() for n in ventanas)]
    if not validos:
        raise RuntimeError("Ningún pluviómetro tiene lluvia válida en toda la ventana.")
    terminadas = sorted(c for c, (cs, _) in cache.items() if cs.index[-1] < horas[-1])
    if terminadas:
        print(f"Pluviómetros cuya serie termina antes del final del evento (excluidos): "
              f"{terminadas}")
    return tablas, validos


def leer_oni() -> pd.Series:
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    fase = oni["ENSO"].astype(str).str.strip().replace({"Neutral": "Neutro"})
    fase.index = oni["date"].dt.normalize()
    return fase[~fase.index.duplicated()]


def leer_eventos(csv: str, crs) -> gpd.GeoDataFrame:
    ev = pd.read_csv(csv)
    ev["fecha_hora"] = pd.to_datetime(ev["fecha_hora"])
    for c in ("lat", "lon"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=["lat", "lon", "fecha_hora"])
    return gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["lon"], ev["lat"]),
                            crs="EPSG:4326").to_crs(crs)


def cargar_capas_contexto(crs) -> dict:
    """Contornos de municipios, comunas de Medellín y río Aburrá."""
    ver = gpd.read_file(MUNI_SHP).to_crs(crs)
    ver["geometry"] = ver.geometry.buffer(0)
    municipios = ver.dissolve("Municipio").boundary
    es_med = ver["Municipio"].astype(str).str.upper().str.contains("MEDELL")
    comunas = ver[es_med].dissolve("Comuna").boundary
    rio = gpd.read_file(RIO_GPKG).to_crs(crs)
    return {"municipios": municipios, "comunas": comunas, "rio": rio}


def _union(geoms):
    return geoms.union_all() if hasattr(geoms, "union_all") else geoms.unary_union


# ============================================================
# MOTOR: clase de lluvia y nivel de alerta de cada SU en una hora
# ============================================================

def evaluar_hora(ts, su, comp, cortes, cfg, lluvia, fase, por_su=True) -> pd.DataFrame:
    """Clase de lluvia (0-3) y nivel de alerta (1-4) por slope unit en la hora ts.

    `lluvia` trae R1, R30 y RLTR: índice ID de SU (o estación si por_su=False).
    Fuera del dominio: FUERA_DOMINIO en ambas. Sin pluviómetro válido: SIN_LLUVIA.
    """
    cod = su[COL_ID] if por_su else su["Codigo_pluvio"]
    R1 = cod.map(lluvia["R1"]).to_numpy(float)
    R30 = cod.map(lluvia["R30"]).to_numpy(float)
    LTR = np.clip(cod.map(lluvia["RLTR"]).to_numpy(float) - R1, 0, None)   # R30 - R1
    dominio = su["dominio"].to_numpy(bool)
    con_lluvia = np.isfinite(R1) & np.isfinite(R30) & np.isfinite(LTR)
    # Filtro de exposición del entrenamiento (mc_common: 1d > 10 y 30d > 50)
    activa = dominio & con_lluvia & (R1 > cfg["UMBRAL_1D"]) & (R30 > cfg["UMBRAL_30D"])

    p = np.full(len(su), np.nan)
    if activa.any():
        z = (comp["b0"] + np.interp(R1[activa], *comp["STR"]) + np.interp(LTR[activa], *comp["LTR"])
             + comp["ENSO"][fase] + su["ef_zona"].to_numpy(float)[activa]
             + su["ef_tipo_su"].to_numpy(float)[activa])
        p[activa] = expit(z)

    clase = np.full(len(su), FUERA_DOMINIO, dtype=int)
    clase[dominio] = 0                                       # Low (incluye bajo el filtro)
    clase[activa] = np.searchsorted(cortes, p[activa], side="right")
    clase[dominio & ~con_lluvia] = SIN_LLUVIA

    tabla = np.array([MATRIZ[c] for c in CLASES_SUSC])      # filas = susceptibilidad
    nivel = np.full(len(su), FUERA_DOMINIO, dtype=int)
    ok = dominio & con_lluvia
    nivel[ok] = tabla[su["susc_idx"].to_numpy()[ok].astype(int), clase[ok]]
    nivel[dominio & ~con_lluvia] = SIN_LLUVIA

    return pd.DataFrame({"R1": R1, "R30": R30, "LTR": LTR, "p": p, "clase": clase, "nivel": nivel},
                        index=su.index)


# ============================================================
# ESTILO DE LOS MAPAS (idéntico al Capítulo 5)
# ============================================================

def fmt_lon(x, _, d=2):
    return f"{abs(x):.{d}f}° {'W' if x < 0 else 'E'}"


def fmt_lat(y, _, d=2):
    return f"{abs(y):.{d}f}° {'S' if y < 0 else 'N'}"


def _decimales(eje, fmt) -> int:
    """Los menos decimales posibles (desde 2, como en el Cap. 5) sin repetir
    etiquetas. En un zoom de 3 km (Granizal) las marcas van cada 0.005° y con
    2 decimales dos marcas distintas saldrían como la misma coordenada."""
    v0, v1 = sorted(eje.get_view_interval())
    ticks = [t for t in eje.get_major_locator().tick_values(v0, v1) if v0 <= t <= v1]
    for d in (2, 3, 4):
        etiquetas = [fmt(t, None, d) for t in ticks]
        if len(set(etiquetas)) == len(etiquetas):
            return d
    return 4


def estilo_mapa(ax, bounds=None, coords=True, nx=3, ny=4, marco=True):
    """Marco fino, marcas hacia adentro, coordenadas abajo y a la izquierda."""
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    for sp in ax.spines.values():
        sp.set_visible(marco)
        sp.set_linewidth(0.9)
        sp.set_edgecolor("black")
    if coords:
        ax.tick_params(direction="in", length=4.5, width=0.9, top=marco, right=marco,
                       labeltop=False, labelright=False, labelbottom=True, labelleft=True,
                       labelsize=7.5)
        ax.xaxis.set_major_locator(MaxNLocator(nx, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(ny, prune="both"))
        dx, dy = _decimales(ax.xaxis, fmt_lon), _decimales(ax.yaxis, fmt_lat)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_lon(v, p, dx)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_lat(v, p, dy)))
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")


def colores_su(valores: np.ndarray, modo: str) -> np.ndarray:
    """Color de cada SU según el nivel de alerta o la clase de lluvia."""
    paleta = COLORES_NIVEL if modo == "alerta" else COLORES_CLASE
    out = np.full(len(valores), COLOR_FUERA, dtype=object)
    for k, c in paleta.items():
        out[valores == k] = c
    out[valores == SIN_LLUVIA] = COLOR_SIN
    return out


def leyenda_mapa(modo: str, hay_sin_lluvia: bool) -> Tuple[list, str]:
    if modo == "alerta":
        h = [Line2D([], [], marker="s", ls="", ms=11, color=COLORES_NIVEL[k],
                    label=NOMBRES_NIVEL[k]) for k in NIVELES]
        titulo = "Alert level"
    else:
        h = [Line2D([], [], marker="s", ls="", ms=11, color=COLORES_CLASE[k], label=n)
             for k, n in enumerate(CLASES_LLUVIA)]
        titulo = "Rainfall class"
    h.append(Line2D([], [], marker="s", ls="", ms=11, color=COLOR_FUERA,
                    label="Outside the analysis domain"))
    if hay_sin_lluvia:
        h.append(Line2D([], [], marker="s", ls="", ms=11, color=COLOR_SIN,
                        label="No valid rainfall estimate"))
    h.append(Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="black",
                    markeredgecolor="white", label="Reported landslide"))
    return h, titulo


class CapaSU:
    """Polígonos de las SU convertidos UNA vez a trayectorias de matplotlib.

    Dibujar 60.914 polígonos con GeoDataFrame.plot en cada panel tarda
    minutos, porque convierte cada geometría en cada llamada. Aquí la
    conversión se hace una sola vez y cada mapa solo cambia los colores.
    Los anillos se orientan (exterior antihorario, huecos horario) para que
    los huecos de las SU se vean como huecos.
    """

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf.reset_index(drop=True)
        self.paths = [self._path(g) for g in self.gdf.geometry]
        self.bounds = tuple(self.gdf.total_bounds)

    @staticmethod
    def _anillo(coords):
        c = np.asarray(coords, float)[:, :2]
        k = np.full(len(c), Path.LINETO, dtype=np.uint8)
        k[0], k[-1] = Path.MOVETO, Path.CLOSEPOLY
        return c, k

    @classmethod
    def _poligono(cls, poly) -> Path:
        poly = orient(poly, sign=1.0)
        partes = [cls._anillo(poly.exterior.coords)] + [cls._anillo(r.coords)
                                                         for r in poly.interiors]
        return Path(np.concatenate([v for v, _ in partes]), np.concatenate([k for _, k in partes]))

    @classmethod
    def _path(cls, g) -> Path:
        if g is None or g.is_empty:
            return Path(np.empty((0, 2)))                    # no dibuja nada
        if g.geom_type == "Polygon":
            return cls._poligono(g)
        return Path.make_compound_path(*[cls._poligono(p) for p in g.geoms if not p.is_empty])

    def indices_en(self, bbox) -> np.ndarray:
        """Posiciones de las SU que tocan un rectángulo (para el zoom)."""
        return np.sort(self.gdf.sindex.query(box(*bbox), predicate="intersects"))

    def dibujar(self, ax, colores=None, valores=None, cmap=None, norm=None, idx=None):
        """Añade las SU al eje. Relleno rasterizado en el PDF (pesa poco y compila rápido)."""
        paths = self.paths if idx is None else [self.paths[i] for i in idx]
        col = PathCollection(paths, linewidths=0, edgecolors="none", rasterized=True, zorder=1)
        if colores is not None:
            col.set_facecolor(colores if idx is None else colores[idx])
        else:
            v = valores if idx is None else valores[idx]
            col.set_array(np.ma.masked_invalid(v))
            col.set_cmap(cmap)
            col.set_norm(norm)
        # La extensión se toma de total_bounds: más rápido que recorrer las
        # 60.914 trayectorias y sin riesgo de que una geometría vacía la altere.
        ax.add_collection(col, autolim=False)
        if idx is None:
            x0, y0, x1, y1 = self.bounds
            ax.update_datalim([(x0, y0), (x1, y1)])
            ax.autoscale_view()
        return col


def _pintar(ax, capa, valores, R1, ocurridos, capas, modo, vmax, ms_ev=26, lw=1.0, idx=None):
    """Contenido de un mapa (valle o zoom): mismas capas en ambos."""
    if modo == "lluvia":
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#eeeeee")
        capa.dibujar(ax, valores=R1, cmap=cmap, norm=plt.Normalize(0, vmax), idx=idx)
    else:
        capa.dibujar(ax, colores=colores_su(valores, modo), idx=idx)
    capas["rio"].plot(ax=ax, color=COLOR_RIO, linewidth=lw, zorder=6)
    capas["comunas"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.45 * lw, alpha=0.75, zorder=7)
    capas["municipios"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.95 * lw, zorder=8)
    if len(ocurridos):
        ocurridos.plot(ax=ax, color="black", edgecolor="white", markersize=ms_ev,
                       linewidth=0.5, zorder=10)


# ============================================================
# FIGURAS DE REJILLA (documento)
# ============================================================

def rejilla_paneles(horas_panel, capa, resultados, ev_d, capas, zb, modo, tag, ev_cfg):
    """Rejilla de paneles: valle completo + zoom de la zona afectada."""
    n = len(horas_panel)
    ncol = 3 if n % 3 == 0 else 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.9 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    vb = capa.bounds
    idx_zoom = capa.indices_en(zb)
    col = "nivel" if modo == "alerta" else "clase"
    vmax = 1.0
    if modo == "lluvia":
        for ts in horas_panel:
            r = resultados[ts]["R1"].dropna()
            if len(r):
                vmax = max(vmax, float(np.nanpercentile(r, 99)))
    inset = ev_cfg.get("inset_pos", INSET_POS)
    conectores = ev_cfg.get("conectores", INSET_CONECTORES)
    hay_sin = False

    for k, ts in enumerate(horas_panel):
        ax = axes[k // ncol][k % ncol]
        res = resultados[ts]
        valores = res[col].to_numpy()
        R1 = res["R1"].to_numpy()
        hay_sin |= bool((valores == SIN_LLUVIA).any())
        ocurridos = ev_d[ev_d["fecha_hora"] <= ts]

        _pintar(ax, capa, valores, R1, ocurridos, capas, modo, vmax, ms_ev=7, lw=0.7)
        estilo_mapa(ax, vb, coords=True, nx=3, ny=4)
        ax.set_aspect("equal")
        ax.set_title(f"({'abcdefgh'[k]})  {ts:%d %b %H:%M}", loc="left", fontsize=10)

        axz = ax.inset_axes(inset)
        _pintar(axz, capa, valores, R1, ocurridos, capas, modo, vmax, ms_ev=24, lw=1.1,
                idx=idx_zoom)
        axz.set_xlim(zb[0], zb[2]); axz.set_ylim(zb[1], zb[3])
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_aspect("equal")
        axz.set_facecolor("white")
        for sp in axz.spines.values():
            sp.set_linewidth(1.1); sp.set_edgecolor("0.15")
        ind = ax.indicate_inset_zoom(axz, edgecolor="0.15", linewidth=1.0, alpha=1.0)
        # matplotlib >= 3.10 devuelve un InsetIndicator; antes, (rectángulo, líneas)
        lineas = ind.connectors if hasattr(ind, "connectors") else ind[1]
        for i, ln in enumerate(lineas):
            ln.set_visible(i in conectores)
            ln.set_linewidth(0.9)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    if modo == "lluvia":
        sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(vmin=0, vmax=vmax))
        cax = fig.add_axes([0.30, 0.045, 0.40, 0.016])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("STR, 24-hour antecedent rainfall (mm)", fontsize=9)
    else:
        handles, titulo = leyenda_mapa(modo, hay_sin)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   ncol=len(handles), frameon=False, fontsize=9.5, title=titulo,
                   title_fontsize=10)

    base = os.path.join(OUT_DIR, f"fig_event_{tag}_{modo}")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", base + ".pdf")


# ============================================================
# FOTOGRAMAS DEL GIF (valle + zoom + hietograma)
# ============================================================

def render_frame(ts, capa, idx_zoom, res, ev_d, ref, capas, ax_map, ax_zoom, ax_rain,
                 ax_acum, zb):
    colores = colores_su(res["nivel"].to_numpy(), "alerta")
    ocurridos = ev_d[ev_d["fecha_hora"] <= ts]
    # Como en el Cap. 5: el valle va sin marco y con la extensión automática de
    # todas las capas (bounds=None); el zoom, con marco y la caja del evento.
    # La relación de aspecto la fija geopandas al dibujar las capas en grados.
    for ax, bounds, msize, marco, idx in ((ax_map, None, 14, False, None),
                                          (ax_zoom, zb, 34, True, idx_zoom)):
        ax.clear()
        capa.dibujar(ax, colores=colores, idx=idx)
        capas["rio"].plot(ax=ax, color=COLOR_RIO, linewidth=1.3, zorder=6)
        capas["municipios"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.9, zorder=7)
        if ax is ax_zoom:
            capas["comunas"].plot(ax=ax, color=COLOR_LINDE, linewidth=0.6, zorder=7)
        if len(ocurridos):
            ocurridos.plot(ax=ax, color="black", edgecolor="white", markersize=msize,
                           linewidth=0.5, zorder=10)
        estilo_mapa(ax, bounds, coords=True, marco=marco)

    ax_map.add_patch(Rectangle((zb[0], zb[1]), zb[2] - zb[0], zb[3] - zb[1],
                               fill=False, edgecolor="black", lw=1.2, zorder=11))
    ax_map.set_title("Aburrá Valley", fontsize=11)
    ax_zoom.set_title(f"Affected area · {ref['zona']}", fontsize=11)

    ax_rain.clear(); ax_acum.clear()
    ax_acum.yaxis.set_label_position("right"); ax_acum.yaxis.tick_right()
    idx, ih, ac = ref["idx"], ref["int_h"], ref["acum"]
    pasado = idx <= ts
    w = FRAME_HOURS / 24 * 0.9
    ax_rain.bar(idx[pasado], ih[pasado], width=w, color=COLOR_INT, zorder=3)
    ax_rain.bar(idx[~pasado], ih[~pasado], width=w, color="#B0BEC5", alpha=0.45, zorder=2)
    ax_acum.plot(idx[pasado], ac[pasado], color=COLOR_ACUM, lw=2, zorder=4)
    ax_acum.plot(idx, ac, color=COLOR_ACUM, lw=0.8, ls=":", alpha=0.5, zorder=3)
    ax_rain.axvline(ts, color="#C62828", lw=1.6, zorder=5)
    t_ev = ref["t_ev"]
    if t_ev is not None and not pd.isna(t_ev) and idx[0] <= t_ev <= idx[-1]:
        ax_rain.axvline(t_ev, color="black", lw=1, ls="--", zorder=4)
        ax_rain.annotate("landslides", xy=(t_ev, 0.97), xycoords=("data", "axes fraction"),
                         ha="right", va="top", fontsize=7.5, rotation=90, clip_on=True)
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
    ax_rain.set_title(f"Reference gauge {ref['cod']} ({ref['dist_km']:.1f} km away)", fontsize=9)
    ax_rain.grid(alpha=0.25, zorder=0)


def hacer_frames(horas, capa, resultados, ev_d, ref, capas, zb, tag, nombre):
    frames_dir = os.path.join(OUT_DIR, f"frames_{tag}")
    os.makedirs(frames_dir, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 7.6), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[2.2, 1],
                          hspace=0.30, wspace=0.18)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_rain = fig.add_subplot(gs[1, 1])
    ax_acum = ax_rain.twinx()
    idx_zoom = capa.indices_en(zb)

    hay_sin = any((resultados[ts]["nivel"] == SIN_LLUVIA).any() for ts in horas)
    handles, titulo = leyenda_mapa("alerta", hay_sin)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(handles), frameon=False, fontsize=9.5, title=titulo,
               title_fontsize=10)

    frames = []
    for i, ts in enumerate(horas):
        render_frame(ts, capa, idx_zoom, resultados[ts], ev_d, ref, capas, ax_map, ax_zoom,
                     ax_rain, ax_acum, zb)
        fig.suptitle(f"{nombre} · {ts:%d %B %Y %H:%M}", fontsize=15, fontweight="bold")
        fpath = os.path.join(frames_dir, f"frame_{i:03d}.png")
        fig.savefig(fpath, bbox_inches="tight")
        frames.append(fpath)
    plt.close(fig)

    import imageio.v3 as iio
    imgs = [iio.imread(f) for f in frames]
    # Todos los fotogramas del mismo tamaño (bbox_inches="tight" puede variar un píxel)
    h = min(im.shape[0] for im in imgs); w = min(im.shape[1] for im in imgs)
    imgs = [im[:h, :w] for im in imgs]
    gif = os.path.join(OUT_DIR, f"{tag}.gif")
    iio.imwrite(gif, imgs, duration=SEG_POR_FRAME * 1000, loop=0)
    try:
        iio.imwrite(os.path.join(OUT_DIR, f"{tag}.mp4"), imgs, fps=int(1 / SEG_POR_FRAME))
    except Exception as e:                                   # sin ffmpeg: solo GIF
        print(f"MP4 no disponible ({type(e).__name__}); queda el GIF.")
    print("GIF:", gif)


# ============================================================
# RESÚMENES Y VALIDACIÓN
# ============================================================

def resumen_por_hora(horas, resultados, su, ev) -> pd.DataFrame:
    dom = su["dominio"].to_numpy(bool)
    filas = []
    for ts in horas:
        r = resultados[ts]
        fila = {"ts": ts, "deslizamientos_reportados": int((ev["fecha_hora"] <= ts).sum()),
                "R1_max_mm": float(np.nanmax(r["R1"])) if r["R1"].notna().any() else np.nan}
        for k in NIVELES:
            fila[f"nivel_{NOMBRES_NIVEL[k]}"] = int((r["nivel"] == k).sum())
        for k, n in enumerate(CLASES_LLUVIA):
            fila[f"clase_{n}"] = int((r["clase"] == k).sum())
        fila["pct_dominio_III_IV"] = round(100 * float((r["nivel"].to_numpy()[dom] >= NIVEL_AVISO)
                                                       .mean()), 2)
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_por_su(horas, resultados, su, ev) -> pd.DataFrame:
    """Por SU del dominio: nivel y clase máximos, primera hora en III-IV, deslizamientos."""
    niv = np.column_stack([resultados[ts]["nivel"].to_numpy() for ts in horas])
    cla = np.column_stack([resultados[ts]["clase"].to_numpy() for ts in horas])
    pmx = np.column_stack([resultados[ts]["p"].to_numpy() for ts in horas])
    horas_idx = pd.DatetimeIndex(horas)
    aviso = niv >= NIVEL_AVISO
    primera = np.where(aviso.any(axis=1), aviso.argmax(axis=1), -1)

    # Deslizamientos del evento dentro de cada SU (en el CRS métrico)
    j = gpd.sjoin(ev[["geometry"]], su[[COL_ID, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]                 # un punto en un borde compartido
    n_ev = j.dropna(subset=[COL_ID]).astype({COL_ID: int}).groupby(COL_ID).size()

    t = pd.DataFrame({
        COL_ID: su[COL_ID].to_numpy(),
        "clase_susc": su["clase_susc_txt"].to_numpy(),
        "dominio": su["dominio"].to_numpy(bool),
        "Codigo_pluvio_referencia": su["Codigo_pluvio"].to_numpy(),
        "nivel_max": niv.max(axis=1),
        "clase_lluvia_max": cla.max(axis=1),
        "p_max": np.nanmax(np.where(np.isfinite(pmx), pmx, -np.inf), axis=1),
        "primera_hora_III_IV": [horas_idx[i] if i >= 0 else pd.NaT for i in primera],
    })
    t.loc[~np.isfinite(t["p_max"]), "p_max"] = np.nan
    t["n_deslizamientos"] = t[COL_ID].map(n_ev).fillna(0).astype(int)
    fuera = j[COL_ID].isna().sum()
    return t, int(fuera), j


def validar(t: pd.DataFrame, ev: gpd.GeoDataFrame, fuera_su: int, tag: str) -> pd.DataFrame:
    """Conteos por SU, como en Métodos: aviso = nivel III o IV en alguna hora."""
    d = t[t["dominio"] & (t["nivel_max"] > SIN_LLUVIA)].copy()
    con = d["n_deslizamientos"] > 0
    filas = []
    for nombre, avisada in (("Rainfall class only", d["clase_lluvia_max"] + 1 >= NIVEL_AVISO),
                            ("Decision matrix", d["nivel_max"] >= NIVEL_AVISO)):
        H = int((con & avisada).sum()); M = int((con & ~avisada).sum())
        FA = int((~con & avisada).sum()); CN = int((~con & ~avisada).sum())
        rec = H / (H + M) if H + M else np.nan
        far = FA / (FA + CN) if FA + CN else np.nan
        filas.append({"evento": tag, "mapas": nombre, "hits": H, "misses": M,
                      "false_alarms": FA, "correct_negatives": CN,
                      "recall": round(rec, 3), "false_alarm_rate": round(far, 4),
                      "HK": round(rec - far, 3)})
    v = pd.DataFrame(filas)
    en_dom = int(t.loc[t["dominio"], "n_deslizamientos"].sum())
    fuera_dom = int(t.loc[~t["dominio"], "n_deslizamientos"].sum())
    print(f"\nValidación ({tag}): {len(ev)} deslizamientos | en SU del dominio: {en_dom} "
          f"en {int(con.sum())} SU | en SU fuera del dominio: {fuera_dom} | fuera de toda SU: "
          f"{fuera_su}")
    print(v.to_string(index=False))
    return v


# ============================================================
# UN EVENTO COMPLETO
# ============================================================

def correr_evento(tag, su, comp, cortes, cfg, cache_series, gpl, fase_oni, capas, zonal):
    ev_cfg = EVENTOS[tag]
    nombre = ev_cfg["nombre"]
    print(f"\n==================== {nombre} ====================")
    if pd.Timestamp(ev_cfg["frame_start"]) < pd.Timestamp(cfg.get("FECHA_CORTE", "2025-04-01")):
        print("AVISO: el evento empieza antes del corte temporal del GAM: no es independiente.")

    horas = pd.date_range(ev_cfg["frame_start"], ev_cfg["frame_end"], freq=f"{FRAME_HOURS}h")

    # --- lluvia por pluviómetro y pluviómetro válido más cercano a cada SU ---
    tablas, validos = lluvia_evento(horas, cache_series, cfg)
    su = mcc.nearest_gauge(su.drop(columns=["Codigo_pluvio", "dist_pluv_m"], errors="ignore"),
                           gpl, set(validos))
    if su["Codigo_pluvio"].isna().any():
        raise RuntimeError("Hay slope units sin pluviómetro válido asignado.")
    su["Codigo_pluvio"] = su["Codigo_pluvio"].astype(int)
    dist = su.loc[su["dominio"], "dist_pluv_m"] / 1000
    print(f"Pluviómetros válidos en toda la ventana: {len(validos)} de {len(cache_series)} | "
          f"distancia SU-pluviómetro (dominio): mediana {dist.median():.1f} km, "
          f"p90 {dist.quantile(.9):.1f} km, máx {dist.max():.1f} km")

    # La estación más cercana se conserva sólo como referencia/comparación.
    tablas_su = interpolar_evento(
        horas, tablas, validos, gpl, su, cfg, COL_ID,
        os.path.join(OUT_DIR, f"lluvia_{tag}"), zonal=zonal,
        guardar_tif=_env("CAP6_GUARDAR_TIF", "1") == "1")

    # --- fase ENSO de cada hora ---
    fases = {}
    for ts in horas:
        f = fase_oni.get(ts.normalize())
        if f is None or f not in comp["ENSO"]:
            raise ValueError(f"Sin fase ENSO válida para {ts} ({f}).")
        fases[ts] = f
    print("Fase ENSO:", sorted(set(fases.values())))

    # --- motor, hora por hora ---
    resultados = {ts: evaluar_hora(ts, su, comp, cortes, cfg,
                                   {n: t.loc[ts] for n, t in tablas_su.items()}, fases[ts])
                  for ts in horas}

    # Recalcular referencia con el MISMO modelo, horas y estaciones válidas.
    referencia = {ts: evaluar_hora(ts, su, comp, cortes, cfg,
                                  {n: t.loc[ts] for n, t in tablas.items()},
                                  fases[ts], por_su=False) for ts in horas}
    cambios = []
    dom = su["dominio"].to_numpy(bool)
    for ts in horas:
        r, b = resultados[ts], referencia[ts]
        cambios.append({"ts": ts,
                        "SU_dominio": int(dom.sum()),
                        "R1_media_kriging_mm": r.loc[dom, "R1"].mean(),
                        "R1_media_vecino_mm": b.loc[dom, "R1"].mean(),
                        "delta_R1_mediana_mm": (r.loc[dom, "R1"] - b.loc[dom, "R1"]).median(),
                        "SU_cambian_clase_lluvia": int((r.loc[dom, "clase"] != b.loc[dom, "clase"]).sum()),
                        "SU_suben_alerta": int((r.loc[dom, "nivel"] > b.loc[dom, "nivel"]).sum()),
                        "SU_bajan_alerta": int((r.loc[dom, "nivel"] < b.loc[dom, "nivel"]).sum())})
    pd.DataFrame(cambios).to_csv(os.path.join(OUT_DIR, f"comparacion_horaria_{tag}.csv"), index=False)

    # --- deslizamientos, zoom y pluviómetro de referencia ---
    ev = leer_eventos(ev_cfg["csv"], su.crs)
    print(f"Deslizamientos del evento: {len(ev)} (reportados el "
          f"{ev['fecha_hora'].min():%d %b %H:%M})")
    x0, y0, x1, y1 = ev.total_bounds
    caja = gpd.GeoSeries.from_xy([x0 - ZOOM_MARGEN_M, x1 + ZOOM_MARGEN_M],
                                 [y0 - ZOOM_MARGEN_M, y1 + ZOOM_MARGEN_M],
                                 crs=su.crs).to_crs(CRS_DIBUJO)
    zb = (caja.x.min(), caja.y.min(), caja.x.max(), caja.y.max())

    gpl_v = gpl[gpl["Codigo"].astype(int).isin(validos)]
    d_pl = gpl_v.geometry.distance(_union(ev.geometry))
    cod_ref = int(gpl_v.loc[d_pl.idxmin(), "Codigo"])
    cs_ref = cache_series[cod_ref][0]
    lluvia_h = cs_ref.diff().reindex(horas).fillna(0.0)
    ref = {"cod": cod_ref, "dist_km": float(d_pl.min()) / 1000.0, "idx": horas,
           "int_h": lluvia_h.to_numpy(float), "acum": lluvia_h.cumsum().to_numpy(float),
           "t_ev": ev["fecha_hora"].min(), "zona": nombre}
    print(f"Pluviómetro de referencia: {cod_ref} a {ref['dist_km']:.1f} km")

    # --- tablas ---
    os.makedirs(OUT_DIR, exist_ok=True)
    resumen = resumen_por_hora(horas, resultados, su, ev)
    resumen.to_csv(os.path.join(OUT_DIR, f"resumen_{tag}.csv"), index=False)
    t, fuera_su, _ = tabla_por_su(horas, resultados, su, ev)
    t.to_csv(os.path.join(OUT_DIR, f"alerta_su_{tag}.csv"), index=False)
    val = validar(t, ev, fuera_su, tag)
    val.to_csv(os.path.join(OUT_DIR, f"validacion_{tag}.csv"), index=False)
    print(f"Máximo del dominio en nivel III-IV durante el evento: "
          f"{resumen['pct_dominio_III_IV'].max():.1f} %")

    t_ref, fuera_ref, _ = tabla_por_su(horas, referencia, su, ev)
    val_ref = validar(t_ref, ev, fuera_ref, tag)
    val_ref.to_csv(os.path.join(OUT_DIR, f"validacion_vecino_{tag}.csv"), index=False)
    pd.concat([val.assign(asignacion="kriging_log_100m"),
               val_ref.assign(asignacion="vecino_mas_cercano")], ignore_index=True).to_csv(
        os.path.join(OUT_DIR, f"comparacion_validacion_{tag}.csv"), index=False)

    # --- figuras (en grados, geometría simplificada solo para dibujar) ---
    su_d = su[["geometry"]].to_crs(CRS_DIBUJO)
    su_d["geometry"] = su_d.geometry.simplify(0.0002)
    capa = CapaSU(su_d)                                      # polígonos -> trayectorias, una vez
    ev_d = ev.to_crs(CRS_DIBUJO)
    if HORAS_PANEL is not None:
        hp = [pd.Timestamp(h) for h in HORAS_PANEL]
    else:
        hp = list(pd.to_datetime(np.linspace(horas[0].value, horas[-1].value, N_PANELES))
                  .floor("h"))
    hp = [h for h in hp if h in resultados]
    print("Horas de los paneles:", [f"{h:%d %b %H:%M}" for h in hp])
    for modo in FIGURAS:
        rejilla_paneles(hp, capa, resultados, ev_d, capas, zb, modo, tag, ev_cfg)
    if HACER_FRAMES:
        hacer_frames(horas, capa, resultados, ev_d, ref, capas, zb, tag, nombre)
    return val


# ============================================================
# MAIN
# ============================================================

def main(eventos: List[str]):
    desconocidos = [e for e in eventos if e not in EVENTOS]
    if desconocidos:
        raise SystemExit(f"Eventos desconocidos: {desconocidos}. Opciones: {list(EVENTOS)}")

    comp, cortes, cfg = cargar_modelo()
    su = cargar_su(comp)
    gpl = mcc.read_pluvios_gdf(su.crs)
    cache_series = cargar_series(gpl["Codigo"].astype(int).tolist())
    gpl = gpl[gpl["Codigo"].astype(int).isin(cache_series)].copy()
    fase_oni = leer_oni()
    capas = cargar_capas_contexto(CRS_DIBUJO)

    print("Preparando intersecciones SU-píxel de 100 m (una sola vez)...", flush=True)
    zonal = ZonalArea(su, cell_size=100, crs="EPSG:32618")
    print(f"Grilla {zonal.shape}; {len(zonal.active_flat)} píxeles con área de SU.", flush=True)
    print(f"mc_common utilizado: {mcc.__file__}")
    vals = [correr_evento(tag, su, comp, cortes, cfg, cache_series, gpl, fase_oni, capas, zonal)
            for tag in eventos]
    if len(vals) > 1:
        pd.concat(vals).to_csv(os.path.join(OUT_DIR, "validacion_eventos.csv"), index=False)
    print("\nListo. Salidas en", OUT_DIR)


if __name__ == "__main__":
    main(sys.argv[1:] or EVENTOS_A_CORRER)
