"""
mc_set_b.py — Paquete B: MISMA fecha, DISTINTA slope unit.
==========================================================

Diseño caso-control apareado por fecha. Para cada evento se muestrean
RATIO_NO_SI slope units de no-deslizamiento EN LA MISMA fecha, con:

    1. Buffer espacial: la slope unit ausencia debe estar a más de
       MIN_DIST_M (200 m) de CUALQUIER slope unit con evento.
    2. Filtro húmedo EN EL SITIO DE LA AUSENCIA: 1d > 10 mm y
       30d > 50 mm en el pluviómetro más cercano a esa slope unit. La
       lluvia varía dentro del valle (puede llover mucho en el sur y
       poco en el norte el mismo día); sin este filtro, parte de la
       separación vendría otra vez de la lluvia, solo que en el
       espacio. Con él, ambas clases recibieron lluvia comparable el
       mismo día y lo que queda por explicar es el terreno.

VARIABLES: estáticas (slope_mean, curva_mean, cobertura, suelos) +
dinámicas (T, P, ENSO). OJO en la interpretación: el rango de T y P
queda estrecho por construcción; su importancia aquí NO mide cuánto
importa la lluvia (esa lectura corresponde al paquete A).

------------------------------------------------------------------
EMPAREJAMIENTO POR DÍA vs POR HORA  (parámetro MATCH_BY_DAY)
------------------------------------------------------------------
El diseño del capítulo dice "la misma FECHA del deslizamiento". Hay
dos formas de implementarlo:

  MATCH_BY_DAY = True  (por defecto, y lo que dice el diseño)
      La ausencia ocurre el mismo día calendario que el evento, en una
      hora en la que su pluviómetro cumple el filtro húmedo. El pool de
      candidatos es amplio.

  MATCH_BY_DAY = False
      La ausencia ocurre en la MISMA HORA exacta que el evento. Es más
      estricto, pero en un valle donde la lluvia es convectiva y local
      casi ningún pluviómetro cumple 1d>10 mm en la hora exacta de un
      evento concreto, y el pool se queda vacío.

Si usas False y el log avisa de muchas fechas sin pool, cambia a True.

------------------------------------------------------------------
NOTA SOBRE UN ERROR YA CORREGIDO
------------------------------------------------------------------
La versión anterior comparaba fechas con
    np.datetime64(t_ev) in set(idx.values)
Eso falla en silencio: numpy considera iguales dos datetime64 de
distinta resolución (ns, us, s) con '==', pero su hash es distinto, y
la pertenencia a un set usa el hash. El resultado era un pool siempre
vacío y un KeyError posterior en add_rain(). Aquí la comparación se
hace con operaciones de pandas sobre DatetimeIndex, que sí manejan
correctamente la resolución temporal.

USO:  python mc_set_b.py
"""

import logging

import numpy as np
import pandas as pd

from mc_common import (CFG, add_rain, apply_wet_filter, build_gauge_cache,
                       load_slope_units, merge_oni, nearest_gauge,
                       read_pluvios_gdf, run_montecarlo, setup_logging,
                       wet_candidates)

TAG = "set_B"

# True  -> la ausencia comparte el DIA del evento (recomendado)
# False -> la ausencia comparte la HORA exacta del evento
MATCH_BY_DAY = True

# --------------------------------------------------------------------
# Preparación (una sola vez)
# --------------------------------------------------------------------
setup_logging(CFG["OUT_DIR"], TAG)
logging.info("MATCH_BY_DAY = %s", MATCH_BY_DAY)

su = load_slope_units()
su_si = su[su[CFG["COL_TARGET"]] == 1].copy().reset_index(drop=True)
su_no = su[su[CFG["COL_TARGET"]] == 0].copy()

# Buffer espacial de 200 m alrededor de TODAS las slope units con evento
geom_si = su_si.geometry.buffer(CFG["MIN_DIST_M"])
buffer_union = (geom_si.union_all() if hasattr(geom_si, "union_all")
                else geom_si.unary_union)
elegibles = (su_no[~su_no.geometry.intersects(buffer_union)]
             .copy().reset_index(drop=True))
logging.info("SU ausencia elegibles (>%.0f m de un evento): %d de %d",
             CFG["MIN_DIST_M"], len(elegibles), len(su_no))

# Pluviómetros: para eventos y para elegibles
gpl = read_pluvios_gdf(su.crs)
cache = build_gauge_cache(gpl["Codigo"].tolist())

su_si = nearest_gauge(su_si, gpl, set(cache.keys()))
su_si = su_si.dropna(subset=["Codigo_pluvio"]).copy()
su_si["Codigo_pluvio"] = su_si["Codigo_pluvio"].astype(int)

elegibles = nearest_gauge(elegibles, gpl, set(cache.keys()))
elegibles = elegibles.dropna(subset=["Codigo_pluvio"]).copy()
elegibles["Codigo_pluvio"] = elegibles["Codigo_pluvio"].astype(int)
logging.info("Distancia SU-pluvio ausencias (m): mediana=%.0f | max=%.0f",
             elegibles["dist_pluv_m"].median(), elegibles["dist_pluv_m"].max())

# Lluvia y filtro húmedo de los eventos
su_si = add_rain(su_si, cache, k_list=[1, 30])
su_si = apply_wet_filter(su_si, "eventos")
su_si = merge_oni(su_si)
su_si["si_no"] = 1
su_si[CFG["COL_FECHA"]] = pd.to_datetime(su_si[CFG["COL_FECHA"]])
logging.info("Eventos tras filtros: %d", len(su_si))

# --------------------------------------------------------------------
# Horas húmedas por pluviómetro, indexadas por la fecha de los eventos
# --------------------------------------------------------------------
cand = wet_candidates(cache, excluded_dates=None)
logging.info("Horas candidatas totales: %d",
             int(sum(len(v) for v in cand.values())))

if MATCH_BY_DAY:
    # dias_evento -> {codigo_pluvio: DatetimeIndex de horas humedas ese dia}
    dias_evento = pd.DatetimeIndex(sorted(
        su_si[CFG["COL_FECHA"]].dt.normalize().unique()))
    wet_by_key = {d: {} for d in dias_evento}

    for cod, idx in cand.items():
        if len(idx) == 0:
            continue
        s = pd.Series(idx.to_numpy(), index=idx.normalize())
        s = s[s.index.isin(dias_evento)]
        if s.empty:
            continue
        for d, grp in s.groupby(level=0):
            wet_by_key[pd.Timestamp(d)][cod] = pd.DatetimeIndex(grp.to_numpy())

    claves = dias_evento
else:
    # hora exacta -> {codigo_pluvio: DatetimeIndex con esa unica hora}
    horas_evento = pd.DatetimeIndex(sorted(
        su_si[CFG["COL_FECHA"]].dt.floor("H").unique()))
    wet_by_key = {t: {} for t in horas_evento}

    for cod, idx in cand.items():
        for t in idx.intersection(horas_evento):
            wet_by_key[pd.Timestamp(t)][cod] = pd.DatetimeIndex([t])

    claves = horas_evento

# --- diagnóstico: tamaño del pool ANTES de lanzar las 100 corridas ----
n_pluv = pd.Series({k: len(v) for k, v in wet_by_key.items()})
n_pool = pd.Series({
    k: int(elegibles["Codigo_pluvio"].isin(v.keys()).sum())
    for k, v in wet_by_key.items()})

logging.info("Pluviometros humedos por fecha de evento: "
             "mediana=%.0f | min=%d | fechas con 0=%d de %d",
             n_pluv.median(), int(n_pluv.min()),
             int((n_pluv == 0).sum()), len(n_pluv))
logging.info("Slope units elegibles por fecha de evento: "
             "mediana=%.0f | min=%d | fechas con menos de %d=%d",
             n_pool.median(), int(n_pool.min()), CFG["RATIO_NO_SI"],
             int((n_pool < CFG["RATIO_NO_SI"]).sum()))

if int((n_pool < CFG["RATIO_NO_SI"]).sum()) > 0.5 * len(n_pool):
    logging.warning(
        "Mas de la mitad de las fechas no tienen pool suficiente. "
        "Si MATCH_BY_DAY=False, cambialo a True. Si ya es True, revisa "
        "los umbrales del filtro humedo o la cobertura de las series.")


# --------------------------------------------------------------------
# Dataset de una corrida Monte Carlo
# --------------------------------------------------------------------
def build_dataset(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    filas = []
    sin_pool = 0

    for _, ev in su_si.iterrows():
        t_ev = pd.Timestamp(ev[CFG["COL_FECHA"]])
        clave = t_ev.normalize() if MATCH_BY_DAY else t_ev.floor("H")

        gauges = wet_by_key.get(clave, {})
        if not gauges:
            sin_pool += 1
            continue

        pool = elegibles[elegibles["Codigo_pluvio"].isin(gauges.keys())]
        if len(pool) < CFG["RATIO_NO_SI"]:
            sin_pool += 1
            continue

        pick = rng.choice(pool.index.values, size=CFG["RATIO_NO_SI"],
                          replace=False)
        for i in pick:
            cod = int(pool.at[i, "Codigo_pluvio"])
            horas = gauges[cod]
            # una hora humeda cualquiera de ese dia (o la hora exacta)
            t_no = pd.Timestamp(rng.choice(horas.to_numpy()))

            fila = pool.loc[i].drop(labels=["geometry"], errors="ignore").to_dict()
            fila[CFG["COL_FECHA"]] = t_no
            fila["si_no"] = 0
            filas.append(fila)

    if sin_pool:
        logging.warning("  %d fechas de evento sin pool suficiente", sin_pool)

    if not filas:
        raise ValueError(
            "No se genero ninguna ausencia en el paquete B. Revisa en el log "
            "las lineas 'Pluviometros humedos por fecha de evento' y "
            "'Slope units elegibles por fecha de evento'. Si casi todas las "
            "fechas tienen 0, prueba MATCH_BY_DAY=True o revisa los umbrales "
            "del filtro humedo.")

    df_no = pd.DataFrame(filas)
    df_no = add_rain(df_no, cache, k_list=[1, 30])
    df_no = apply_wet_filter(df_no, "ausencias")   # confirma la regla 2
    df_no = merge_oni(df_no)

    df = pd.concat([su_si.drop(columns="geometry", errors="ignore"), df_no],
                   ignore_index=True)
    df["T"] = df["1d"].astype(float)
    df["P"] = (df["30d"].astype(float) - df["1d"].astype(float)).clip(lower=0)

    logging.info("  dataset: SI=%d | NO=%d | total=%d",
                 int((df.si_no == 1).sum()), int((df.si_no == 0).sum()), len(df))
    return df


# --------------------------------------------------------------------
# Monte Carlo: estáticas + dinámicas
# --------------------------------------------------------------------
if __name__ == "__main__":
    run_montecarlo(TAG, build_dataset,
                   num_cols=["T", "P", "slope_mean", "curva_mean"],
                   cat_cols=["cobertura", "suelos", "ENSO"])