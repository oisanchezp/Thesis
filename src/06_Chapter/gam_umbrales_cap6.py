"""
gam_umbrales_cap6.py
====================

Componente temporal del Capítulo 6: GAM logístico y umbrales
probabilísticos de lluvia, con el diseño de muestreo C del Capítulo 5.

    logit(p) = α + s1(STR1) + s2(LTR30) + γ_ENSO + δ_zona + η_tipo

Reemplaza a gam_umbrales_pooled.py. Reutiliza mc_common.py del Capítulo 5
(lluvia, pluviómetros, ONI, filtro húmedo, métricas y cortes por TPR), de
modo que las reglas de lluvia son exactamente las del conjunto C.

QUÉ CAMBIA FRENTE A gam_umbrales_pooled.py
------------------------------------------
                    gam_umbrales_pooled.py          este script
  Dominio           su_susceptibles_con_            su_susceptibles_parte2.gpkg
                    inventario.gpkg (modelo viejo)  (dominio del ZINB + BYM2 actual)
  Eventos           columnas del gpkg viejo         inventario_unificado.csv, unido
                                                    por ubicación a las SU del dominio
  Corte temporal    solo el rango de la tabla ONI   eventos y ausencias < FECHA_CORTE
  Presencias        una fila por registro           una por slope unit y día
  Fecha ausencia    a ±30 días de cualquier evento  a ±EXCL_DAYS_C días de cualquier
                                                    evento (10 por defecto) y, si
                                                    MATCH_TIME, con el año-mes de los
                                                    eventos (desactivado por defecto)
  Filtro húmedo     >= 10 mm y >= 50 mm             > 10 mm y > 50 mm (mc_common)
  Lluvia            huecos = 0, duplicados sum()    huecos = NaN con cobertura >= 90 %,
                                                    duplicados max() (mc_common)
  Pluviómetro       el más cercano ya instalado     el más cercano (mc_common)
  Partición         70/30 estratificada, sin        80/20 agrupada por slope unit y
                    agrupar, semilla fija 42        estratificada, semilla de la corrida
  Lambda            70/30 aleatorio                 70/30 agrupado dentro del train
  Salidas           grilla densa por corrida        efectos aditivos por corrida (la
                                                    grilla se reconstruye exacta)

REGLAS DEL MUESTREO DE AUSENCIAS
-------------------------------
  1. La slope unit de la ausencia está a más de 200 m de toda slope unit
     con un deslizamiento registrado antes del corte.
  2. La fecha de la ausencia está a más de EXCL_DAYS_C días de TODA fecha
     con un registro en el inventario (ventana d - k ... d + k vetada).
  3. Filtro húmedo en ambas clases, en el pluviómetro de cada unidad:
     R1 > 10 mm y R30 > 50 mm.
  4. (opcional, MATCH_TIME) Las ausencias imitan la distribución año-mes de
     las presencias; si está apagada, la fecha sale al azar entre TODAS las
     horas húmedas del pluviómetro, anteriores al corte.
  Relación 2:1, 100 corridas Monte Carlo.

  Configuraciones (cada una escribe en su propia subcarpeta de Umbrales/):
      excl10d_libre   EXCL_DAYS_C = 10, MATCH_TIME = False   <- por defecto
      excl0d_anomes   EXCL_DAYS_C = 0,  MATCH_TIME = True    (set C del Cap. 5)
  Se cambian arriba o con CAP6_EXCL_DAYS=0 CAP6_MATCH_TIME=1 python ...

DECISIONES QUE SE FIJAN ARRIBA (y deben coincidir con Métodos)
-------------------------------------------------------------
  LTR_DEF     "R30-R1" = lo que calcularon los modelos del conjunto C;
              "R31-R1" = la definición del barrido de ventanas.
  N_SPLINES   8 (Métodos del Cap. 6); el Cap. 5 usó 6.
  DEDUP       "su_dia": varios registros en la misma unidad y el mismo día
              son UNA observación (se conserva la hora más temprana).
  SPLIT       "estratificado_agrupado": 80/20 por slope unit, con la misma
              proporción de clases (lo que dice el texto del Cap. 5);
              "agrupado" = GroupShuffleSplit, lo que corrió el Cap. 5.

SALIDAS (OUT_DIR)
-----------------
  gam_cap6_mc_metrics.csv       métricas en el test de cada corrida
  gam_cap6_mc_thresholds.csv    P*95, P*85, P*70 y Youden por corrida
  gam_cap6_mc_roc.csv           ROC interpolada sobre una rejilla común de FPR
  gam_cap6_mc_effects.csv       efectos parciales por corrida (logit, sin centrar)
  gam_cap6_mc_test_pred.csv     (y, p) del test de cada corrida -> calibración
  gam_cap6_mc_levels.csv        observaciones por nivel de ENSO, zona y tipo
  gam_cap6_mc_temporal.csv      presencias y ausencias por año-mes y corrida
  gam_cap6_mc_ausencias.csv     todas las ausencias muestreadas
  gam_cap6_mc_lambda.csv        AUROC de validación de cada lambda
  gam_cap6_eventos.csv          presencias de entrenamiento (fijas en todas las corridas)
  gam_cap6_eventos_descartados.csv   registros descartados y motivo
  gam_cap6_eventos_validacion.csv    registros del dominio desde FECHA_CORTE
  gam_cap6_flujo_eventos.csv    registros que sobreviven a cada paso
  gam_cap6_dataset_run_star.csv datos de la corrida representativa
  gam_cap6_efectos_final.csv    efectos del modelo final, con IC 95 %
  gam_cap6_modelo_final.joblib  modelo final (corrida de AUROC mediana,
                                reajustado con toda su muestra) + mapeos
  gam_cap6_resumen.json         configuración, conteos y medianas
  gam_cap6.log

USO
---
    python gam_umbrales_cap6.py
Correr desde cualquier carpeta; las rutas son absolutas. Cada ruta se puede
cambiar arriba o con la variable de entorno indicada (útil para pruebas).
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from pygam import LogisticGAM, f, s
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold     # scikit-learn >= 1.0
except ImportError:                                               # pragma: no cover
    StratifiedGroupKFold = None


# ============================================================
# CONFIGURACIÓN
# ============================================================

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


CH5_DIR = _env("CAP6_CH5_DIR", "/home/oisanchezp/Thesis/src/05_Chapter")   # mc_common.py
CH6_DIR = _env("CAP6_CH6_DIR", "/home/oisanchezp/Thesis/src/06_Chapter")

PATH_DOMINIO   = _env("CAP6_PATH_DOMINIO", f"{CH6_DIR}/salidas/su_susceptibles_parte2.gpkg")
LAYER_DOMINIO  = "susceptibles"
PATH_SU_TODAS  = _env("CAP6_PATH_SU_TODAS", f"{CH6_DIR}/salidas/su_resultados_ZINB_BYM2.gpkg")
LAYER_SU_TODAS = "susceptibilidad"          # solo para diagnosticar eventos fuera del dominio
# AJUSTA esta ruta: el inventario fechado (496 registros)
PATH_INVENTARIO = _env("CAP6_PATH_INVENTARIO",
                       "/home/oisanchezp/Thesis/data/processed/inventario_unificado.csv")

PLUV_META   = _env("CAP6_PLUV_META",
                   "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv")
RUTA_SERIES = _env("CAP6_RUTA_SERIES",
                   "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/")
ONI_PATH    = _env("CAP6_ONI_PATH",
                   "/home/oisanchezp/Thesis/data/metadata/oni_diario_2010_2025.csv")
TAG         = "gam_cap6"

# --- diseño temporal de las ausencias (reglas 2 y 4) ---
EXCL_DAYS_C = int(_env("CAP6_EXCL_DAYS", "10"))      # ±días vetados alrededor de cada deslizamiento
MATCH_TIME  = _env("CAP6_MATCH_TIME", "0") == "1"    # True = año-mes de los deslizamientos
DISENO_TEMPORAL = f"excl{EXCL_DAYS_C}d_" + ("anomes" if MATCH_TIME else "libre")
OUT_DIR     = _env("CAP6_OUT_DIR", f"{CH6_DIR}/Umbrales/{DISENO_TEMPORAL}/")

# --- resto del diseño, igual al set C del Cap. 5 (se fija aquí aunque mc_common cambie) ---
DISENO = {
    "UMBRAL_1D":   10.0,     # filtro húmedo: R1 > 10 mm
    "UMBRAL_30D":  50.0,     #                R30 > 50 mm
    "MIN_DIST_M":  200.0,    # regla 1
    "RATIO_NO_SI": 2,        # ausencias : presencias
    "N_MC":        int(_env("CAP6_N_MC", "100")),
    "TEST_SIZE":   0.20,     # 80/20
    "BASE_SEED":   100,
    "MIN_COV":     0.90,     # cobertura mínima de datos en cada ventana
    "H1":          24,
    "LAM_GRID":    np.logspace(0, 3, 5),
    "TPR_TARGETS": (0.70, 0.85, 0.95),
}

# Validación temporal: todo lo que entra al modelo es anterior a esta
# fecha; los registros posteriores (Altavista, Olivares) se guardan aparte.
FECHA_CORTE = pd.Timestamp(_env("CAP6_FECHA_CORTE", "2025-04-01"))

LTR_DEF   = "R30-R1"                 # "R30-R1" | "R31-R1"
N_SPLINES = 8
DEDUP     = "su_dia"                 # "su_dia" | "su_hora" | None
SPLIT     = "estratificado_agrupado" # | "agrupado"
LAMBDA_VAL_SIZE = 0.30               # partición interna para elegir lambda

# --- columnas y factores ---
COL_FECHA = "fecha_hora_evento"
ENSO_NIVELES = ["La Niña", "Neutro", "El Niño"]
FACTORES = ["ENSO", "zona", "tipo_su"]
FEATURES = ["STR", "LTR"] + FACTORES

# --- rejillas en las que se guardan los efectos (mm) ---
STR_GRID = np.arange(0.0, 300.0 + 1e-9, 1.0)     # cubre los extremos observados
LTR_GRID = np.arange(0.0, 1000.0 + 1e-9, 5.0)
ROC_FPR_GRID = np.linspace(0.0, 1.0, 101)

K_LIST = [1, 30, 31] if LTR_DEF == "R31-R1" else [1, 30]
K_COLS = [f"{k}d" for k in K_LIST]


# ============================================================
# mc_common (Capítulo 5)
# ============================================================

_NECESARIAS = ["CFG", "add_rain", "apply_wet_filter", "build_gauge_cache",
               "eval_metrics", "excluded_dates_global", "merge_oni",
               "nearest_gauge", "read_pluvios_gdf", "setup_logging",
               "thresholds_at_tpr", "wet_candidates"]

if not os.path.isfile(os.path.join(CH5_DIR, "mc_common.py")):
    raise FileNotFoundError(
        f"No encuentro mc_common.py en {CH5_DIR}. Ajusta CH5_DIR al inicio del script.")
if CH5_DIR not in sys.path:
    sys.path.insert(0, CH5_DIR)
import mc_common as mcc  # noqa: E402

_faltan = [n for n in _NECESARIAS if not hasattr(mcc, n)]
if _faltan:
    raise ImportError(f"mc_common.py no tiene {_faltan}; ¿es la versión del conjunto C?")

CFG = mcc.CFG
_CFG_PREVIO = {k: CFG.get(k) for k in DISENO}
CFG.update(DISENO)
CFG.update({"PLUV_META": PLUV_META, "RUTA_SERIES": RUTA_SERIES,
            "ONI_PATH": ONI_PATH, "OUT_DIR": OUT_DIR, "COL_FECHA": COL_FECHA,
            # la versión nueva de mc_common corta las candidatas en HOLDOUT_START:
            # se alinea con FECHA_CORTE para que los dos cortes sean el mismo
            "HOLDOUT_START": str(FECHA_CORTE.date())})


# ============================================================
# UTILIDADES
# ============================================================

def nivel(v) -> str:
    """'2', 2, 2.0 y '2.0' -> '2'; el texto se deja igual. NaN -> None."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    try:
        x = float(v)
        return str(int(x)) if x.is_integer() else str(v).strip()
    except (TypeError, ValueError):
        return str(v).strip()


def paso(flujo: List[Dict], texto: str, df: pd.DataFrame) -> None:
    """Registra cuántos registros (y slope units) sobreviven a un paso."""
    n_su = int(df["su_id"].nunique()) if "su_id" in df.columns else None
    flujo.append({"paso": texto, "registros": int(len(df)), "slope_units": n_su})
    logging.info("  %-58s %5d registros%s", texto, len(df),
                 "" if n_su is None else f" | {n_su} SU")


def descartar(desc: List[pd.DataFrame], df: pd.DataFrame, motivo: str) -> None:
    if len(df):
        d = df.drop(columns="geometry", errors="ignore").copy()
        d["motivo"] = motivo
        desc.append(d)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """STR1 = R1; LTR30 = R30 - R1 (o R31 - R1, según LTR_DEF)."""
    df = df.copy()
    df["STR"] = df["1d"].astype(float)
    largo = "31d" if LTR_DEF == "R31-R1" else "30d"
    df["LTR"] = (df[largo].astype(float) - df["1d"].astype(float)).clip(lower=0.0)
    return df


# ============================================================
# 1) DOMINIO Y EVENTOS
# ============================================================

def cargar_dominio() -> gpd.GeoDataFrame:
    """Slope units del dominio del componente espacial (incluir_parte2)."""
    su = gpd.read_file(PATH_DOMINIO, layer=LAYER_DOMINIO)
    faltan = [c for c in ["su_id", "zona", "tipo_su"] if c not in su.columns]
    if faltan:
        raise KeyError(f"Faltan {faltan} en {PATH_DOMINIO}. Hay: {sorted(su.columns)}")
    if su.crs is None:
        raise ValueError("El dominio no tiene CRS.")
    if su.crs.is_geographic:
        su = su.to_crs("EPSG:32618")
    if "incluir_parte2" in su.columns and not su["incluir_parte2"].astype(bool).all():
        logging.warning("El archivo del dominio trae unidades con incluir_parte2 = False; "
                        "se conservan solo las True.")
        su = su[su["incluir_parte2"].astype(bool)].copy()

    su["su_id"] = su["su_id"].astype(int)
    if su["su_id"].duplicated().any():
        raise ValueError("su_id repetido en el dominio.")
    for c in ["zona", "tipo_su"]:
        su[c] = su[c].map(nivel)
        if su[c].isna().any():
            raise ValueError(f"{c} tiene valores vacíos en el dominio.")
    su["su_uid"] = su["su_id"]          # grupo del split
    su = su.reset_index(drop=True)
    logging.info("Dominio: %d slope units | CRS %s | zonas %s | tipos %s",
                 len(su), su.crs.to_string(), sorted(su["zona"].unique()),
                 sorted(su["tipo_su"].unique()))
    return su


def cargar_eventos(su: gpd.GeoDataFrame, flujo: List[Dict], desc: List[pd.DataFrame]):
    """Registros fechados -> presencias candidatas (una por SU y día).

    Devuelve (ev, validacion, fechas_inventario):
        ev                 : registros de entrenamiento en el dominio, con su_id
        validacion         : registros del dominio desde FECHA_CORTE
        fechas_inventario  : fechas de TODOS los registros (regla 2)
    """
    inv = pd.read_csv(PATH_INVENTARIO)
    for c in [COL_FECHA, "latitud", "longitud"]:
        if c not in inv.columns:
            raise KeyError(f"Falta la columna '{c}' en {PATH_INVENTARIO}.")
    # nombres que chocarían con los del modelo
    inv = inv.rename(columns={"si_no": "si_no_inventario", "Codigo": "Codigo_inventario"})
    inv["id_registro"] = np.arange(len(inv))
    inv[COL_FECHA] = pd.to_datetime(inv[COL_FECHA], errors="coerce")
    paso(flujo, "Registros del inventario", inv)

    malos = inv[COL_FECHA].isna() | inv["latitud"].isna() | inv["longitud"].isna()
    descartar(desc, inv[malos], "sin fecha o sin coordenadas")
    inv = inv[~malos].copy()
    fechas_inventario = inv[COL_FECHA].copy()

    pts = gpd.GeoDataFrame(inv, geometry=gpd.points_from_xy(inv["longitud"], inv["latitud"]),
                           crs="EPSG:4326").to_crs(su.crs)
    j = gpd.sjoin(pts, su[["su_id", "geometry"]], how="left", predicate="intersects")
    j = j[~j.index.duplicated(keep="first")].drop(columns="index_right", errors="ignore")

    fuera = j["su_id"].isna()
    if fuera.any():
        motivo = pd.Series("fuera de las slope units modeladas", index=j.index[fuera])
        if os.path.isfile(PATH_SU_TODAS):
            try:
                todas = gpd.read_file(PATH_SU_TODAS, layer=LAYER_SU_TODAS, columns=["su_id"])
            except Exception:                       # motor fiona sin 'columns'
                todas = gpd.read_file(PATH_SU_TODAS, layer=LAYER_SU_TODAS)[["su_id", "geometry"]]
            todas = todas.to_crs(su.crs)
            k = gpd.sjoin(j.loc[fuera, ["geometry"]], todas[["su_id", "geometry"]],
                          how="inner", predicate="intersects")
            motivo.loc[motivo.index.isin(k.index)] = "en una slope unit excluida del dominio"
        for m in motivo.unique():
            descartar(desc, j.loc[motivo.index[motivo == m]], m)
    j = j[~fuera].copy()
    j["su_id"] = j["su_id"].astype(int)
    paso(flujo, "Dentro del dominio de análisis", j)

    validacion = j[j[COL_FECHA] >= FECHA_CORTE].drop(columns="geometry").copy()
    descartar(desc, j[j[COL_FECHA] >= FECHA_CORTE],
              f"posterior al corte ({FECHA_CORTE.date()}): validación")
    ev = j[j[COL_FECHA] < FECHA_CORTE].drop(columns="geometry").copy()
    paso(flujo, f"Anteriores al {FECHA_CORTE.date()}", ev)

    if DEDUP:
        ev = ev.sort_values([COL_FECHA, "id_registro"])
        clave = (ev[COL_FECHA].dt.normalize() if DEDUP == "su_dia"
                 else ev[COL_FECHA].dt.floor("h"))
        dup = pd.DataFrame({"su": ev["su_id"], "t": clave}).duplicated(keep="first")
        descartar(desc, ev[dup.to_numpy()], "repetido en la misma slope unit y "
                  + ("día" if DEDUP == "su_dia" else "hora"))
        ev = ev[~dup.to_numpy()].copy()
        paso(flujo, "Una presencia por slope unit y " + ("día" if DEDUP == "su_dia" else "hora"), ev)
    return ev.reset_index(drop=True), validacion.reset_index(drop=True), fechas_inventario


def lluvia_eventos(ev: pd.DataFrame, su: gpd.GeoDataFrame, gpl, cache,
                   flujo: List[Dict], desc: List[pd.DataFrame]) -> pd.DataFrame:
    """Pluviómetro, lluvia, filtro húmedo y ENSO de las presencias (como mc_set_c)."""
    cols_su = [c for c in ["su_id", "su_uid", "zona", "tipo_su", "clase_susc", "lambda", "mm",
                           "geometry"] if c in su.columns]
    evg = su[cols_su].merge(ev, on="su_id", how="inner").reset_index(drop=True)

    evg = mcc.nearest_gauge(evg, gpl, set(cache.keys()))
    sin = evg["Codigo_pluvio"].isna()
    descartar(desc, evg[sin], "sin pluviómetro con serie")
    evg = evg[~sin].copy()
    evg["Codigo_pluvio"] = evg["Codigo_pluvio"].astype(int)
    paso(flujo, "Con pluviómetro asignado", evg)

    evg = mcc.add_rain(evg.reset_index(drop=True), cache, k_list=K_LIST)
    incompleta = evg[K_COLS].isna().any(axis=1)
    descartar(desc, evg[incompleta], "serie incompleta en la ventana (cobertura < 90 %)")
    evg = evg[~incompleta].copy()
    paso(flujo, "Con lluvia completa en las ventanas", evg)

    humedo = (evg["1d"] > CFG["UMBRAL_1D"]) & (evg["30d"] > CFG["UMBRAL_30D"])
    descartar(desc, evg[~humedo], "bajo el filtro de exposición")
    evg = mcc.apply_wet_filter(evg, "eventos")          # mismo criterio, misma función
    paso(flujo, f"Sobre el filtro (R1 > {CFG['UMBRAL_1D']:g} mm y "
                f"R30 > {CFG['UMBRAL_30D']:g} mm)", evg)

    evg = mcc.merge_oni(evg)
    sin_enso = evg["ENSO"].isna()
    descartar(desc, evg[sin_enso], "sin fase ENSO (fecha fuera de la tabla ONI)")
    evg = evg[~sin_enso].copy()
    paso(flujo, "Con fase ENSO", evg)

    evg["ENSO"] = evg["ENSO"].astype(str)
    evg["si_no"] = 1
    evg = add_features(evg)
    return pd.DataFrame(evg.drop(columns="geometry", errors="ignore")).reset_index(drop=True)


# ============================================================
# 2) POOL DE AUSENCIAS (reglas 1, 2, 3 y 4)
# ============================================================

def preparar_ausencias(su: gpd.GeoDataFrame, su_con_evento: np.ndarray,
                       fechas_inventario: pd.Series, ev: pd.DataFrame, gpl, cache,
                       resumen: Dict):
    """Slope units elegibles y horas candidatas por pluviómetro y año-mes."""
    # Regla 1: buffer de 200 m alrededor de las SU con deslizamiento (antes del corte)
    con_ev = su["su_id"].isin(su_con_evento)
    buf = gpd.GeoDataFrame(geometry=su.loc[con_ev, "geometry"].buffer(CFG["MIN_DIST_M"]).values,
                           crs=su.crs)
    no = su.loc[~con_ev]
    tocan = gpd.sjoin(no[["geometry"]], buf, how="inner", predicate="intersects").index.unique()
    elegibles = no.drop(index=tocan).copy()
    logging.info("SU ausencia elegibles (>%.0f m de una SU con evento): %d de %d",
                 CFG["MIN_DIST_M"], len(elegibles), len(no))

    elegibles = mcc.nearest_gauge(elegibles, gpl, set(cache.keys()))
    elegibles = elegibles.dropna(subset=["Codigo_pluvio"]).copy()
    elegibles["Codigo_pluvio"] = elegibles["Codigo_pluvio"].astype(int)

    # Reglas 2 y 3: horas húmedas a más de EXCL_DAYS_C días de todo registro del inventario
    excl = mcc.excluded_dates_global(fechas_inventario, EXCL_DAYS_C)
    cand = mcc.wet_candidates(cache, excluded_dates=excl)
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    t_min, t_max = oni["date"].min(), oni["date"].max() + pd.Timedelta(hours=23)
    fin = min(FECHA_CORTE, t_max)
    cand = {c: i[(i >= t_min) & (i < fin)] for c, i in cand.items()}
    cand = {c: i for c, i in cand.items() if len(i)}
    logging.info("Dias vetados (±%d d de cada registro): %d | ONI %s a %s | horas candidatas: %d",
                 EXCL_DAYS_C, len(excl), t_min.date(), oni["date"].max().date(),
                 int(sum(len(v) for v in cand.values())))

    if MATCH_TIME:
        # Regla 4: las horas de cada pluviómetro se agrupan por año-mes y solo se
        # guardan los meses con deslizamientos, que se sortean con su peso
        peso = ev[COL_FECHA].dt.to_period("M").value_counts(normalize=True).sort_index()
        cand_por = {}
        for cod, idx in cand.items():
            per = idx.to_period("M")
            m = per.isin(peso.index)
            if m.any():
                g = pd.Series(idx[m], index=per[m]).groupby(level=0)
                cand_por[cod] = {p: pd.DatetimeIndex(h.to_numpy()) for p, h in g}
        cobertura = pd.Series({c: len(d) for c, d in cand_por.items()}, dtype=float)
        logging.info("Regla 4 activa | periodos con eventos: %d (%s a %s) | pluviómetros con "
                     "candidatas en esos periodos: %d | periodos cubiertos: mediana %.0f",
                     len(peso), peso.index.min(), peso.index.max(), len(cand_por),
                     cobertura.median() if len(cobertura) else 0)
    else:
        # Sin regla 4: cualquier hora húmeda del registro del pluviómetro
        peso = None
        cand_por = dict(cand)
        anhos = pd.Series(np.concatenate([v.year.to_numpy() for v in cand_por.values()]))
        logging.info("Regla 4 apagada | pluviómetros con candidatas: %d | horas candidatas por "
                     "año: %s", len(cand_por), anhos.value_counts().sort_index().to_dict())

    pool = elegibles[elegibles["Codigo_pluvio"].isin(cand_por.keys())]
    cols = [c for c in ["su_id", "su_uid", "zona", "tipo_su", "clase_susc", "lambda", "mm",
                        "Codigo_pluvio", "dist_pluv_m"] if c in pool.columns]
    pool = pd.DataFrame(pool[cols]).reset_index(drop=True)

    n_no = CFG["RATIO_NO_SI"] * len(ev)
    logging.info("Pool de SU ausencia utilizable: %d (se necesitan %d por corrida)",
                 len(pool), n_no)
    if len(pool) < n_no:
        raise ValueError(f"Pool insuficiente: {len(pool)} < {n_no}.")

    resumen.update({
        "su_elegibles_200m": int(len(elegibles)), "su_pool": int(len(pool)),
        "dias_vetados": int(len(excl)),
        "periodos_con_eventos": int(len(peso)) if peso is not None else None,
        "pluviometros_con_candidatas": int(len(cand_por)),
        "horas_candidatas": int(sum(len(v) for v in cand.values())),
        "dist_pluvio_pool_m": {"mediana": float(pool["dist_pluv_m"].median()),
                               "p95": float(pool["dist_pluv_m"].quantile(0.95)),
                               "max": float(pool["dist_pluv_m"].max())},
    })
    return pool, cand_por, peso


def construir_ausencias(seed: int, pool: pd.DataFrame, cand_por: Dict, peso,
                        n_no: int, cache) -> Tuple[pd.DataFrame, Dict]:
    """Ausencias de una corrida (como mc_set_c.build_dataset).

    cand_por[cod] es un DatetimeIndex con todas las horas candidatas (regla 4
    apagada) o un dict año-mes -> horas (regla 4 activa).
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool.index.to_numpy(), size=n_no, replace=False)
    filas, sin_fecha = [], 0
    for i in pick:
        row = pool.loc[i]
        d = cand_por[int(row["Codigo_pluvio"])]
        if MATCH_TIME:
            ps = [p for p in peso.index if p in d]
            if not ps:
                sin_fecha += 1
                continue
            w = np.array([peso[p] for p in ps], dtype=float)
            horas = d[ps[int(rng.choice(len(ps), p=w / w.sum()))]]
        else:
            horas = d
        fila = row.to_dict()
        fila[COL_FECHA] = pd.Timestamp(rng.choice(horas.to_numpy()))
        filas.append(fila)

    no = mcc.add_rain(pd.DataFrame(filas).reset_index(drop=True), cache, k_list=K_LIST)
    n0 = len(no)
    no = no.dropna(subset=K_COLS)
    n_nan = n0 - len(no)
    no = mcc.apply_wet_filter(no, "ausencias")            # defensivo
    no = mcc.merge_oni(no)
    n_sin_enso = int(no["ENSO"].isna().sum())
    no = no.dropna(subset=["ENSO"]).copy()
    no["ENSO"] = no["ENSO"].astype(str)
    no["si_no"] = 0
    info = {"ausencias_sorteadas": int(n_no), "sin_fecha": int(sin_fecha),
            "sin_lluvia": int(n_nan), "sin_enso": n_sin_enso, "ausencias": int(len(no))}
    return add_features(no), info


# ============================================================
# 3) MODELO
# ============================================================

def mapas_categorias(su: gpd.GeoDataFrame) -> Dict[str, Dict[str, int]]:
    """Códigos fijos para todas las corridas (el orden lo da el dominio)."""
    orden = lambda vals: sorted(vals, key=lambda v: (float(v) if v.replace('.', '', 1).isdigit()
                                                     else np.inf, v))
    return {"ENSO": {v: i for i, v in enumerate(ENSO_NIVELES)},
            "zona": {v: i for i, v in enumerate(orden(su["zona"].unique()))},
            "tipo_su": {v: i for i, v in enumerate(orden(su["tipo_su"].unique()))}}


def matriz_X(df: pd.DataFrame, mapas: Dict) -> np.ndarray:
    cols = [df["STR"].to_numpy(float), df["LTR"].to_numpy(float)]
    for v in FACTORES:
        cod = df[v].astype(str).map(mapas[v])
        if cod.isna().any():
            raise ValueError(f"Niveles de {v} sin código: {sorted(df.loc[cod.isna(), v].unique())}")
        cols.append(cod.to_numpy(float))
    return np.column_stack(cols)


def terminos():
    return (s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES)
            + f(2) + f(3) + f(4))


def niveles_completos(X: np.ndarray, mapas: Dict) -> bool:
    """pyGAM asigna en silencio un nivel ausente del train al vecino: se exige que estén todos."""
    return all(set(np.unique(X[:, j]).astype(int)) == set(mapas[v].values())
               for j, v in enumerate(FACTORES, start=2))


def particion(y: np.ndarray, grupos: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """80/20 agrupado por slope unit (y estratificado por clase, si SPLIT lo pide)."""
    if SPLIT == "estratificado_agrupado":
        if StratifiedGroupKFold is None:
            raise ImportError("StratifiedGroupKFold pide scikit-learn >= 1.0; "
                              "actualízalo o usa SPLIT = 'agrupado'.")
        k = int(round(1.0 / CFG["TEST_SIZE"]))
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        itr, ite = next(cv.split(np.zeros(len(y)), y, groups=grupos))
    else:
        cv = GroupShuffleSplit(n_splits=1, test_size=CFG["TEST_SIZE"], random_state=seed)
        itr, ite = next(cv.split(np.zeros(len(y)), y, groups=grupos))
    assert not set(grupos[itr]) & set(grupos[ite]), "una SU cayó a los dos lados"
    return itr, ite


def elegir_lambda(X: np.ndarray, y: np.ndarray, grupos: np.ndarray, mapas: Dict,
                  seed: int) -> Tuple[float, List[Dict]]:
    """Lambda con mayor AUROC en una partición 70/30 del train, agrupada por SU."""
    for intento in range(20):
        cv = GroupShuffleSplit(n_splits=1, test_size=LAMBDA_VAL_SIZE, random_state=seed + 1000 * intento)
        a, b = next(cv.split(X, y, groups=grupos))
        if niveles_completos(X[a], mapas) and len(np.unique(y[b])) == 2:
            break
    else:
        raise ValueError("No hubo partición interna con todos los niveles en el train.")
    mejor, tabla = None, []
    for lam in CFG["LAM_GRID"]:
        g = LogisticGAM(terminos(), lam=float(lam)).fit(X[a], y[a])
        auc = float(roc_auc_score(y[b], g.predict_proba(X[b])))
        tabla.append({"lam": float(lam), "auc_val": auc})
        if mejor is None or auc > mejor[1]:
            mejor = (float(lam), auc)
    return mejor[0], tabla


def efectos(gam: LogisticGAM, mapas: Dict) -> pd.DataFrame:
    """Contribución de cada término en la escala logit, con su IC 95 % (pyGAM).

    logit(p) = intercepto + s1(STR) + s2(LTR) + γ + δ + η, exactamente, así
    que la probabilidad en cualquier punto de las rejillas se reconstruye
    sumando filas de esta tabla. pyGAM no centra las curvas: la constante
    queda repartida entre el intercepto y los términos.
    """
    filas = [{"term": "intercept", "x": np.nan, "level": "", "effect": float(gam.coef_[-1]),
              "lo": np.nan, "hi": np.nan}]
    m = len(FEATURES)
    for j, (nom, grid) in enumerate([("STR", STR_GRID), ("LTR", LTR_GRID)]):
        XX = np.zeros((len(grid), m))
        XX[:, j] = grid
        pdep, ci = gam.partial_dependence(term=j, X=XX, width=0.95)
        filas += [{"term": nom, "x": float(x), "level": "", "effect": float(e),
                   "lo": float(c[0]), "hi": float(c[1])} for x, e, c in zip(grid, pdep, ci)]
    for j, v in enumerate(FACTORES, start=2):
        inv = {c: k for k, c in mapas[v].items()}
        codes = np.array(sorted(inv), dtype=float)
        XX = np.zeros((len(codes), m))
        XX[:, j] = codes
        pdep, ci = gam.partial_dependence(term=j, X=XX, width=0.95)
        filas += [{"term": v, "x": float(c), "level": inv[int(c)], "effect": float(e),
                   "lo": float(ci_[0]), "hi": float(ci_[1])} for c, e, ci_ in zip(codes, pdep, ci)]
    return pd.DataFrame(filas)


def centros(gam: LogisticGAM, X: np.ndarray) -> Dict[str, float]:
    """Media de s1 y s2 sobre las observaciones: para dibujar las curvas centradas."""
    return {"centro_STR": float(np.mean(gam.partial_dependence(term=0, X=X))),
            "centro_LTR": float(np.mean(gam.partial_dependence(term=1, X=X)))}


def correr(df: pd.DataFrame, run_id: int, seed: int, mapas: Dict) -> Dict:
    """Una corrida: partición, lambda, ajuste, métricas, cortes, ROC y efectos."""
    df = df.dropna(subset=["si_no"] + FEATURES).reset_index(drop=True)
    y = df["si_no"].astype(int).to_numpy()
    X = matriz_X(df, mapas)
    grupos = df["su_uid"].to_numpy()

    itr, ite = particion(y, grupos, seed)
    if not niveles_completos(X[itr], mapas):
        raise ValueError("un nivel de ENSO, zona o tipo no aparece en el train")
    lam, tabla_lam = elegir_lambda(X[itr], y[itr], grupos[itr], mapas, seed)
    gam = LogisticGAM(terminos(), lam=lam).fit(X[itr], y[itr])

    p = np.asarray(gam.predict_proba(X[ite]))
    met = mcc.eval_metrics(y[ite], p)
    thr = mcc.thresholds_at_tpr(y[ite], p)
    fpr, tpr, _ = roc_curve(y[ite], p)
    tpr_i = np.interp(ROC_FPR_GRID, fpr, tpr)
    tpr_i[0], tpr_i[-1] = 0.0, 1.0

    met.update({"run_id": run_id, "seed": seed, "best_lam": lam,
                "edof": float(gam.statistics_["edof"]),
                "n": len(y), "n_si": int(y.sum()), "n_no": int((y == 0).sum()),
                "n_train": len(itr), "n_test": len(ite), "n_test_si": int(y[ite].sum()),
                "su_train": int(len(np.unique(grupos[itr]))),
                "su_test": int(len(np.unique(grupos[ite]))),
                **centros(gam, X[itr])})

    pred = df.loc[ite, ["su_id", COL_FECHA, "STR", "LTR", *FACTORES]].copy()
    pred.insert(0, "run_id", run_id)
    pred["y"] = y[ite]
    pred["p"] = p
    return {"metrics": met, "gam": gam, "lam": lam, "df": df,
            "thresholds": [{"run_id": run_id, "thr_level": k, **v} for k, v in thr.items()],
            "roc": pd.DataFrame({"run_id": run_id, "model": "GAM",
                                 "fpr": ROC_FPR_GRID, "tpr": tpr_i}),
            "effects": efectos(gam, mapas).assign(run_id=run_id),
            "pred": pred,
            "lambda": pd.DataFrame(tabla_lam).assign(run_id=run_id)}


# ============================================================
# 4) MAIN
# ============================================================

def main() -> None:
    t0 = time.time()
    mcc.setup_logging(OUT_DIR, TAG)
    logging.info("GAM del Capitulo 6 | ausencias: ±%d d de cada registro, año-mes %s (%s) | "
                 "LTR = %s | %d splines | split %s | dedup %s | corte %s",
                 EXCL_DAYS_C, "sí" if MATCH_TIME else "no", DISENO_TEMPORAL, LTR_DEF,
                 N_SPLINES, SPLIT, DEDUP, FECHA_CORTE.date())
    cambios = {k: (v, DISENO[k]) for k, v in _CFG_PREVIO.items()
               if not np.array_equal(np.asarray(v, dtype=object), np.asarray(DISENO[k], dtype=object))}
    if cambios:
        logging.warning("Valores de mc_common reemplazados por los del diseño C: %s", cambios)

    flujo, desc, resumen = [], [], {}

    # --- 1. dominio, eventos y lluvia ---------------------------------------
    su = cargar_dominio()
    mapas = mapas_categorias(su)
    logging.info("Codigos: %s", mapas)

    logging.info("Flujo de los registros del inventario:")
    ev, validacion, fechas_inv = cargar_eventos(su, flujo, desc)
    su_con_evento = ev["su_id"].unique()        # regla 1: toda SU con registro antes del corte

    gpl = mcc.read_pluvios_gdf(su.crs)
    cache = mcc.build_gauge_cache(gpl["Codigo"].tolist())
    ev = lluvia_eventos(ev, su, gpl, cache, flujo, desc)
    if ev.empty:
        raise RuntimeError("No quedó ninguna presencia.")
    resumen["dist_pluvio_eventos_m"] = {"mediana": float(ev["dist_pluv_m"].median()),
                                        "p95": float(ev["dist_pluv_m"].quantile(0.95)),
                                        "max": float(ev["dist_pluv_m"].max())}
    logging.info("Presencias finales: %d en %d SU | por zona %s | por ENSO %s | por tipo %s",
                 len(ev), ev["su_id"].nunique(), ev["zona"].value_counts().to_dict(),
                 ev["ENSO"].value_counts().to_dict(), ev["tipo_su"].value_counts().to_dict())

    # --- 2. pool de ausencias -----------------------------------------------
    pool, cand_por, peso = preparar_ausencias(su, su_con_evento, fechas_inv, ev, gpl, cache, resumen)
    n_no = CFG["RATIO_NO_SI"] * len(ev)

    # --- 3. Monte Carlo -------------------------------------------------------
    out = {k: [] for k in ["metrics", "thresholds", "roc", "effects", "pred", "lambda",
                           "levels", "temporal", "ausencias"]}
    datasets, info_runs = {}, []
    for mc in range(CFG["N_MC"]):
        seed = CFG["BASE_SEED"] + mc
        logging.info("=== MC %d/%d | seed=%d ===", mc + 1, CFG["N_MC"], seed)
        try:
            no, info = construir_ausencias(seed, pool, cand_por, peso, n_no, cache)
            df = pd.concat([ev, no], ignore_index=True)
            r = correr(df, mc, seed, mapas)
        except Exception as e:                       # como run_montecarlo del Cap. 5
            logging.exception("run %d falló: %s", mc, e)
            continue

        info_runs.append({"run_id": mc, **info})
        datasets[mc] = r["df"]
        out["metrics"].append(r["metrics"])
        out["thresholds"] += r["thresholds"]
        for k in ["roc", "effects", "pred", "lambda"]:
            out[k].append(r[k])
        d = r["df"]
        for v in FACTORES:
            out["levels"].append(d.groupby([v, "si_no"]).size().rename("n").reset_index()
                                 .rename(columns={v: "level"}).assign(variable=v, run_id=mc))
        out["temporal"].append(d.assign(periodo=d[COL_FECHA].dt.to_period("M").astype(str))
                               .groupby(["si_no", "periodo"]).size().rename("n").reset_index()
                               .assign(run_id=mc))
        cols_no = [c for c in ["su_id", COL_FECHA, *K_COLS, "STR", "LTR", "ONI", *FACTORES,
                               "clase_susc", "Codigo_pluvio", "dist_pluv_m"] if c in d.columns]
        out["ausencias"].append(d.loc[d["si_no"] == 0, cols_no].assign(run_id=mc))
        m = r["metrics"]
        pc = {t["thr_level"]: t["threshold"] for t in r["thresholds"]}
        logging.info("  lam=%g | AUROC=%.3f AP=%.3f Brier=%.3f HK=%.3f | P*95=%.3f P*85=%.3f "
                     "P*70=%.3f", r["lam"], m["auc"], m["ap"], m["brier"], m["hk"],
                     pc["P_TPR95"], pc["P_TPR85"], pc["P_TPR70"])

    if not out["metrics"]:
        raise RuntimeError("No hubo corridas exitosas.")

    # --- 4. salidas por corrida ---------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    P = lambda n: os.path.join(OUT_DIR, f"{TAG}_{n}")
    met = pd.DataFrame(out["metrics"])
    met = met[["run_id", "seed"] + [c for c in met.columns if c not in ("run_id", "seed")]]
    met = met.merge(pd.DataFrame(info_runs), on="run_id", how="left")
    met.to_csv(P("mc_metrics.csv"), index=False)
    thr = pd.DataFrame(out["thresholds"])
    thr.to_csv(P("mc_thresholds.csv"), index=False)
    pd.concat(out["roc"]).to_csv(P("mc_roc.csv"), index=False)
    pd.concat(out["effects"]).to_csv(P("mc_effects.csv"), index=False)
    pd.concat(out["pred"]).to_csv(P("mc_test_pred.csv"), index=False)
    pd.concat(out["lambda"]).to_csv(P("mc_lambda.csv"), index=False)
    pd.concat(out["levels"]).to_csv(P("mc_levels.csv"), index=False)
    pd.concat(out["temporal"]).to_csv(P("mc_temporal.csv"), index=False)
    pd.concat(out["ausencias"]).to_csv(P("mc_ausencias.csv"), index=False)

    ev.to_csv(P("eventos.csv"), index=False)
    validacion.to_csv(P("eventos_validacion.csv"), index=False)
    (pd.concat(desc, ignore_index=True) if desc else pd.DataFrame()).to_csv(
        P("eventos_descartados.csv"), index=False)
    pd.DataFrame(flujo).to_csv(P("flujo_eventos.csv"), index=False)

    # --- 5. modelo final: corrida de AUROC mediana, reajustada con toda su muestra
    med = float(met["auc"].median())
    star = int(met.loc[(met["auc"] - med).abs().idxmin(), "run_id"])
    lam_star = float(met.loc[met["run_id"] == star, "best_lam"].iloc[0])
    df_star = datasets[star]
    X_star = matriz_X(df_star, mapas)
    y_star = df_star["si_no"].astype(int).to_numpy()
    gam_final = LogisticGAM(terminos(), lam=lam_star).fit(X_star, y_star)
    ef_final = efectos(gam_final, mapas)
    ef_final.to_csv(P("efectos_final.csv"), index=False)
    df_star.to_csv(P("dataset_run_star.csv"), index=False)

    cortes = thr.groupby("thr_level")[["threshold", "tpr", "fpr", "tnr"]].median()
    info_final = {"run_star": star, "seed_star": CFG["BASE_SEED"] + star, "lam": lam_star,
                  "edof": float(gam_final.statistics_["edof"]), "n": int(len(y_star)),
                  "n_si": int(y_star.sum()), **centros(gam_final, X_star)}
    joblib.dump({"gam": gam_final, "mapas": mapas, "features": FEATURES,
                 "ltr_def": LTR_DEF, "n_splines": N_SPLINES,
                 "cortes_mediana": cortes.to_dict("index"),
                 "fecha_corte": str(FECHA_CORTE.date()), **info_final},
                P("modelo_final.joblib"))
    resumen["modelo_final"] = info_final

    q = lambda c: {"mediana": float(met[c].median()), "p05": float(met[c].quantile(0.05)),
                   "p95": float(met[c].quantile(0.95))}
    resumen.update({
        "config": {**{k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v)
                      for k, v in DISENO.items()},
                   "EXCL_DAYS_C": EXCL_DAYS_C, "MATCH_TIME": MATCH_TIME,
                   "DISENO_TEMPORAL": DISENO_TEMPORAL,
                   "FECHA_CORTE": str(FECHA_CORTE.date()), "LTR_DEF": LTR_DEF,
                   "N_SPLINES": N_SPLINES, "DEDUP": DEDUP, "SPLIT": SPLIT,
                   "LAMBDA_VAL_SIZE": LAMBDA_VAL_SIZE,
                   "STR_GRID": [float(STR_GRID[0]), float(STR_GRID[-1]), len(STR_GRID)],
                   "LTR_GRID": [float(LTR_GRID[0]), float(LTR_GRID[-1]), len(LTR_GRID)],
                   "PATH_DOMINIO": PATH_DOMINIO, "PATH_INVENTARIO": PATH_INVENTARIO},
        "mapas": mapas,
        "presencias": int(len(ev)), "su_con_presencia": int(ev["su_id"].nunique()),
        "registros_validacion": int(len(validacion)),
        "corridas_exitosas": int(len(met)), "run_star": star, "lam_star": lam_star,
        "metricas": {c: q(c) for c in ["auc", "ap", "brier", "hk", "recall", "precision",
                                       "pofd", "f1"]},
        "cortes_mediana": cortes.round(4).to_dict("index"),
        "minutos": round((time.time() - t0) / 60, 1),
    })
    with open(P("resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2, default=str)

    logging.info("Corridas exitosas: %d de %d | run estrella: %d (lam=%g)",
                 len(met), CFG["N_MC"], star, lam_star)
    logging.info("AUROC %.3f [%.3f-%.3f] | AP %.3f | Brier %.3f | HK %.3f",
                 med, met["auc"].quantile(.05), met["auc"].quantile(.95),
                 met["ap"].median(), met["brier"].median(), met["hk"].median())
    logging.info("Cortes (mediana):\n%s", cortes.round(3).to_string())
    logging.info("=== LISTO en %.1f min | salidas en %s ===", (time.time() - t0) / 60, OUT_DIR)


if __name__ == "__main__":
    main()