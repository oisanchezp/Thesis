"""
mc_set_a.py — Paquete A: MISMA slope unit, DISTINTA fecha.
==========================================================

Diseño caso-control apareado por slope unit. Para cada evento se
muestrean RATIO_NO_SI fechas de no-deslizamiento EN LA MISMA slope
unit, con estas reglas:

    1. La fecha debe estar a más de ±EXCL_DAYS (30) días de CUALQUIER
       evento registrado en ESA slope unit (exclusión POR UNIDAD, no
       global: una exclusión global vaciaría las temporadas de lluvia
       de 2021-2025 y las ausencias vendrían de otra época del año).
    2. La fecha debe ser húmeda: 1d > 10 mm y 30d > 50 mm en el
       pluviómetro más cercano a la slope unit (el mismo criterio del
       inventario -> las dos clases quedan condicionadas igual).

VARIABLES DEL MODELO: solo las dinámicas (T='1d', P='30d'-'1d', ENSO).
Las estáticas son idénticas en ambas clases por construcción (misma
slope unit), no discriminan y NO deben entrar ni interpretarse aquí.

USO:  python mc_set_a.py
SALIDAS (en CFG['OUT_DIR']): set_A_mc_metrics.csv, set_A_mc_thresholds.csv,
set_A_mc_importance.csv, set_A.log
"""

import logging

import numpy as np
import pandas as pd

from mc_common import (CFG, add_rain, apply_wet_filter, build_gauge_cache,
                       excluded_dates_by_su, load_slope_units, merge_oni,
                       nearest_gauge, read_pluvios_gdf, run_montecarlo,
                       setup_logging, wet_candidates)

TAG = "set_A"

# --------------------------------------------------------------------
# Preparación (una sola vez, fuera del bucle Monte Carlo)
# --------------------------------------------------------------------
setup_logging(CFG["OUT_DIR"], TAG)

su = load_slope_units()
su_si = su[su[CFG["COL_TARGET"]] == 1].copy().reset_index(drop=True)

# Pluviómetro más cercano a cada slope unit con evento
gpl = read_pluvios_gdf(su.crs)
cache = build_gauge_cache(gpl["Codigo"].tolist())
su_si = nearest_gauge(su_si, gpl, set(cache.keys()))
su_si = su_si.dropna(subset=["Codigo_pluvio"]).copy()
su_si["Codigo_pluvio"] = su_si["Codigo_pluvio"].astype(int)
logging.info("Distancia SU-pluvio (m): mediana=%.0f | max=%.0f",
             su_si["dist_pluv_m"].median(), su_si["dist_pluv_m"].max())

# Lluvia de los eventos + filtro húmedo simétrico
su_si = add_rain(su_si, cache, k_list=[1, 30])
su_si = su_si.rename(columns={"1d": "1d", "30d": "30d"})
su_si = apply_wet_filter(su_si.rename(columns={}), "eventos")
su_si = merge_oni(su_si)
su_si["si_no"] = 1
logging.info("Eventos tras filtros: %d", len(su_si))

# Fechas húmedas candidatas por pluviómetro (sin exclusión global:
# la exclusión temporal del paquete A es por slope unit)
cand = wet_candidates(cache, excluded_dates=None)

# Exclusión de ±30 días alrededor de los eventos de cada slope unit
excl_su = excluded_dates_by_su(su_si, CFG["EXCL_DAYS"])


# --------------------------------------------------------------------
# Dataset de una corrida Monte Carlo
# --------------------------------------------------------------------
def build_dataset(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    filas = []
    for _, ev in su_si.iterrows():
        cod = int(ev["Codigo_pluvio"])
        fechas = cand.get(cod)
        if fechas is None:
            continue
        # quitar las fechas vetadas de ESTA slope unit
        veto = excl_su.get(ev["su_uid"])
        if veto is not None and len(veto):
            fechas = fechas[~pd.Index(fechas.normalize().date).isin(veto)]
        if len(fechas) < CFG["RATIO_NO_SI"]:
            continue
        pick = rng.choice(fechas.values, size=CFG["RATIO_NO_SI"], replace=False)
        for t in pick:
            fila = ev.drop(labels=["geometry"], errors="ignore").to_dict()
            fila[CFG["COL_FECHA"]] = pd.Timestamp(t)
            fila["si_no"] = 0
            filas.append(fila)

    df_no = pd.DataFrame(filas)
    df_no = add_rain(df_no, cache, k_list=[1, 30])   # lluvia de SU fecha nueva
    df_no = apply_wet_filter(df_no, "ausencias")      # defensivo (ya cumplen)
    df_no = merge_oni(df_no)

    df = pd.concat([su_si.drop(columns="geometry", errors="ignore"), df_no],
                   ignore_index=True)
    # T y P sin solapamiento: P = lluvia 30d - lluvia 1d
    df["T"] = df["1d"].astype(float)
    df["P"] = (df["30d"].astype(float) - df["1d"].astype(float)).clip(lower=0)
    return df


# --------------------------------------------------------------------
# Monte Carlo: solo variables dinámicas
# --------------------------------------------------------------------
if __name__ == "__main__":
    run_montecarlo(TAG, build_dataset,
                   num_cols=["T", "P"],
                   cat_cols=["ENSO"])
