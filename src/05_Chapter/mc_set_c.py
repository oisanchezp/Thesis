"""
mc_set_c.py — Paquete C: DISTINTA slope unit y DISTINTA fecha.
==============================================================

Diseño caso-control NO apareado (el de Steger et al. 2024 y Moreno et
al. 2024). Es el único paquete donde estáticas y dinámicas se estiman
juntas, y el que sostiene el modelo final y los umbrales.

Reglas para las ausencias:
    1. Buffer espacial: slope unit a más de MIN_DIST_M (200 m) de
       cualquier slope unit con evento.
    2. Fecha distinta de la de cualquier evento del inventario.
    3. Filtro húmedo en el sitio y la fecha de la ausencia:
       1d > 10 mm y 30d > 50 mm.
    4. La distribución de años y meses de las ausencias imita la de las
       presencias (muestreo estratificado en el tiempo).

------------------------------------------------------------------
POR QUÉ CAMBIÓ LA REGLA 2 (leer antes de comparar con versiones viejas)
------------------------------------------------------------------
La versión anterior excluía ±30 días alrededor de CADA fecha de evento.
Con ~479 eventos concentrados en 2021-2025 y en las dos temporadas de
lluvia, esa unión de ventanas borraba prácticamente todas las
temporadas lluviosas de esos años. Las ausencias sobrevivientes caían
sobre todo en 2012-2020, así que el modelo comparaba presencias de un
periodo contra ausencias de otro y podía separarlas por ÉPOCA en vez de
por proceso. ENSO era el caso más claro: 2015-16 fue El Niño fuerte y
2021-22 La Niña, de modo que la fase habría separado las clases sin
significar nada físico.

Además, la exclusión temporal amplia no tiene sentido en este paquete:
las slope units elegibles están a más de 200 m de cualquier evento, es
decir, NUNCA tuvieron un deslizamiento, así que no hay "fecha de evento
en esa unidad" de la que alejarse. Y excluir los días en que hubo
deslizamientos en otro sector del valle es contraproducente, porque una
ladera estable en un día que sí disparó deslizamientos ajenos es
justamente una ausencia informativa.

La regla queda entonces en lo que el diseño pide de verdad -fecha
distinta- y el sesgo temporal se controla con la regla 4, que es la
estrategia de Steger et al. (2024): equilibrar las ausencias por año y
por mes en lugar de vetar periodos completos.

Si quieres recuperar el comportamiento anterior, sube EXCL_DAYS_C.

USO:  python mc_set_c.py
"""

import logging

import numpy as np
import pandas as pd

from mc_common import (CFG, add_rain, apply_wet_filter, build_gauge_cache,
                       excluded_dates_global, load_slope_units, merge_oni,
                       nearest_gauge, read_pluvios_gdf, run_montecarlo,
                       setup_logging, wet_candidates)

TAG = "set_C"

# Días excluidos alrededor de cada evento. 0 = solo el día exacto del
# evento (lo que pide el diseño: "fecha distinta"). Subirlo reintroduce
# el sesgo temporal descrito arriba.
EXCL_DAYS_C = 0

# True  -> las ausencias imitan la distribución año-mes de las presencias
# False -> fecha uniforme entre todas las candidatas (sesgo temporal)
MATCH_TIME = True

# --------------------------------------------------------------------
# Preparación (una sola vez)
# --------------------------------------------------------------------
setup_logging(CFG["OUT_DIR"], TAG)
logging.info("EXCL_DAYS_C = %d | MATCH_TIME = %s", EXCL_DAYS_C, MATCH_TIME)

su = load_slope_units()
su_si = su[su[CFG["COL_TARGET"]] == 1].copy().reset_index(drop=True)
su_no = su[su[CFG["COL_TARGET"]] == 0].copy()

# Regla 1: buffer espacial de 200 m
geom_si = su_si.geometry.buffer(CFG["MIN_DIST_M"])
buffer_union = (geom_si.union_all() if hasattr(geom_si, "union_all")
                else geom_si.unary_union)
elegibles = (su_no[~su_no.geometry.intersects(buffer_union)]
             .copy().reset_index(drop=True))
logging.info("SU ausencia elegibles (>%.0f m de un evento): %d de %d",
             CFG["MIN_DIST_M"], len(elegibles), len(su_no))

# Pluviómetros
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

# Reglas 2 y 3: fechas húmedas, quitando los días de evento
excl = excluded_dates_global(su_si[CFG["COL_FECHA"]], EXCL_DAYS_C)
logging.info("Dias excluidos: %d", len(excl))
cand = wet_candidates(cache, excluded_dates=excl)
logging.info("Horas candidatas totales: %d",
             int(sum(len(v) for v in cand.values())))

n_no_target = CFG["RATIO_NO_SI"] * len(su_si)

# --------------------------------------------------------------------
# Regla 4: distribución objetivo año-mes, tomada de las presencias
# --------------------------------------------------------------------
periodos_si = su_si[CFG["COL_FECHA"]].dt.to_period("M")
peso = periodos_si.value_counts(normalize=True).sort_index()
logging.info("Periodos (ano-mes) con eventos: %d | rango %s a %s",
             len(peso), str(peso.index.min()), str(peso.index.max()))

# Por pluviómetro: periodo -> horas húmedas de ese periodo.
# Solo se guardan los periodos en los que hubo eventos, que son los
# únicos de los que se va a muestrear.
cand_per = {}
for cod, idx in cand.items():
    if len(idx) == 0:
        continue
    per = idx.to_period("M")
    d = {}
    for p in peso.index:
        m = (per == p)
        if m.any():
            d[p] = idx[m]
    if d:
        cand_per[cod] = d

cobertura = pd.Series({cod: len(d) for cod, d in cand_per.items()})
logging.info("Pluviometros con candidatas en periodos de evento: %d",
             len(cand_per))
logging.info("Periodos cubiertos por pluviometro: mediana=%.0f de %d",
             cobertura.median() if len(cobertura) else 0, len(peso))

# Pool de slope units cuyo pluviómetro tiene candidatas utilizables
pool_base = elegibles[elegibles["Codigo_pluvio"].isin(cand_per.keys())]
logging.info("Pool de SU ausencia utilizable: %d (se necesitan %d)",
             len(pool_base), n_no_target)
if len(pool_base) < n_no_target:
    raise ValueError(
        f"Pool insuficiente: {len(pool_base)} < {n_no_target}. "
        "Revisa el filtro humedo o la cobertura de las series.")


# --------------------------------------------------------------------
# Dataset de una corrida Monte Carlo
# --------------------------------------------------------------------
def build_dataset(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pick = rng.choice(pool_base.index.values, size=n_no_target, replace=False)
    filas = []
    sin_fecha = 0

    for i in pick:
        row = pool_base.loc[i]
        d = cand_per[int(row["Codigo_pluvio"])]

        if MATCH_TIME:
            # periodo sorteado con la distribucion de las presencias,
            # restringido a los periodos que ese pluviometro cubre
            ps = [p for p in peso.index if p in d]
            if not ps:
                sin_fecha += 1
                continue
            w = np.array([peso[p] for p in ps], dtype=float)
            w = w / w.sum()
            p_sel = ps[int(rng.choice(len(ps), p=w))]
            horas = d[p_sel]
        else:
            horas = pd.DatetimeIndex(np.concatenate(
                [v.to_numpy() for v in d.values()]))

        t = pd.Timestamp(rng.choice(horas.to_numpy()))

        fila = row.drop(labels=["geometry"], errors="ignore").to_dict()
        fila[CFG["COL_FECHA"]] = t
        fila["si_no"] = 0
        filas.append(fila)

    if sin_fecha:
        logging.warning("  %d SU descartadas por no tener fecha candidata "
                        "en los periodos de los eventos", sin_fecha)

    if not filas:
        raise ValueError(
            "No se genero ninguna ausencia en el paquete C. Revisa en el log "
            "'Pool de SU ausencia utilizable' y 'Horas candidatas totales'.")

    df_no = pd.DataFrame(filas)
    df_no = add_rain(df_no, cache, k_list=[1, 30])
    df_no = apply_wet_filter(df_no, "ausencias")   # defensivo
    df_no = merge_oni(df_no)

    df = pd.concat([su_si.drop(columns="geometry", errors="ignore"), df_no],
                   ignore_index=True)
    df["T"] = df["1d"].astype(float)
    df["P"] = (df["30d"].astype(float) - df["1d"].astype(float)).clip(lower=0)

    # --- control del equilibrio temporal (esto es un RESULTADO) -------
    a_si = pd.to_datetime(df.loc[df.si_no == 1, CFG["COL_FECHA"]]).dt.year
    a_no = pd.to_datetime(df.loc[df.si_no == 0, CFG["COL_FECHA"]]).dt.year
    logging.info("  anos presencias: %s", a_si.value_counts().sort_index().to_dict())
    logging.info("  anos ausencias : %s", a_no.value_counts().sort_index().to_dict())

    m_si = pd.to_datetime(df.loc[df.si_no == 1, CFG["COL_FECHA"]]).dt.month
    m_no = pd.to_datetime(df.loc[df.si_no == 0, CFG["COL_FECHA"]]).dt.month
    logging.info("  meses presencias: %s", m_si.value_counts().sort_index().to_dict())
    logging.info("  meses ausencias : %s", m_no.value_counts().sort_index().to_dict())

    logging.info("  dataset: SI=%d | NO=%d | total=%d",
                 int((df.si_no == 1).sum()), int((df.si_no == 0).sum()), len(df))
    return df


# --------------------------------------------------------------------
# Monte Carlo: estáticas + dinámicas (modelo completo)
# --------------------------------------------------------------------
if __name__ == "__main__":
    run_montecarlo(TAG, build_dataset,
                   num_cols=["T", "P", "slope_mean", "curva_mean"],
                   cat_cols=["cobertura", "suelos", "ENSO"])