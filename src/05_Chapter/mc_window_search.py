"""
mc_window_search.py — Barrido de ventanas STR (1-7 d) x LTR (8-90 d).
=====================================================================

Selecciona la mejor combinación de días de lluvia de corto (STR) y
largo plazo (LTR) usando el PAQUETE A (misma slope unit, distinta
fecha): con el terreno controlado por construcción, la ventana ganadora
refleja lluvia y no confusión con las condiciones del sitio.

PROCEDIMIENTO
-------------
1. Por corrida MC (semilla nueva -> ausencias nuevas del paquete A):
   a. A cada observación (evento o ausencia) se le calcula UNA VEZ el
      perfil de acumulados 1..97 días (rain_profile), con control de
      cobertura de datos. Esto evita recalcular lluvia por combinación.
   b. Para cada (t, p) con t=1..7 y p=8..90 (t+p <= 97):
         STR = acum[t] ;  LTR = acum[t+p] - acum[t]
      se ajusta un GAM(STR, LTR, ENSO) sobre el 80% de entrenamiento
      y se evalúa AUROC en el 20% de test (held-out, no CV).
2. Se agregan las N_MC corridas: mediana y desviación del AUROC por
   combinación. La ganadora es la de mayor MEDIANA de AUROC.

El criterio es AUROC y no HK/F1 porque estos últimos dependen del corte
0.5, que con la relación 2:1 no es neutro; el AUROC no depende de él.

SALIDAS: window_search_raw.csv (todas las corridas) y
window_search_summary.csv (mediana/media/std por combinación).

USO:  python mc_window_search.py
NOTA DE COSTO: 581 combinaciones x N_MC corridas. Con N_MC=100 son
~58,000 ajustes de GAM. Ajusta N_MC_SWEEP abajo si necesitas un ensayo
rápido (p. ej. 10) antes de la corrida definitiva.
"""

import logging
import os

import numpy as np
import pandas as pd
from pygam import LogisticGAM, f, s
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mc_common import (CFG, ENSO_MAP, build_gauge_cache, excluded_dates_by_su,
                       load_slope_units, merge_oni, nearest_gauge,
                       rain_profile, read_pluvios_gdf, setup_logging,
                       wet_candidates)

TAG = "window_search"
N_MC_SWEEP = CFG["N_MC"]        # bajar a 10 para ensayos
MAX_DAYS   = 97
SHORT      = range(1, 8)        # STR: 1..7 días
LONG       = range(8, 91)       # LTR: 8..90 días

# --------------------------------------------------------------------
# Preparación: igual que el paquete A
# --------------------------------------------------------------------
setup_logging(CFG["OUT_DIR"], TAG)

su = load_slope_units()
su_si = su[su[CFG["COL_TARGET"]] == 1].copy().reset_index(drop=True)

gpl = read_pluvios_gdf(su.crs)
cache = build_gauge_cache(gpl["Codigo"].tolist())
su_si = nearest_gauge(su_si, gpl, set(cache.keys()))
su_si = su_si.dropna(subset=["Codigo_pluvio"]).copy()
su_si["Codigo_pluvio"] = su_si["Codigo_pluvio"].astype(int)
su_si = merge_oni(su_si)
su_si["si_no"] = 1

cand = wet_candidates(cache, excluded_dates=None)
excl_su = excluded_dates_by_su(su_si, CFG["EXCL_DAYS"])

TERMS = s(0, n_splines=CFG["N_SPLINES"]) + s(1, n_splines=CFG["N_SPLINES"]) + f(2)


def profile_for(df: pd.DataFrame) -> np.ndarray:
    """Matriz (n_obs x MAX_DAYS) de acumulados 1..97 d por observación."""
    M = np.full((len(df), MAX_DAYS), np.nan)
    for r, (_, row) in enumerate(df.iterrows()):
        cs, cv = cache[int(row["Codigo_pluvio"])]
        M[r] = rain_profile(cs, cv, row[CFG["COL_FECHA"]], MAX_DAYS)
    return M


def sample_absences(seed: int) -> pd.DataFrame:
    """Ausencias del paquete A (misma SU, distinta fecha, húmeda)."""
    rng = np.random.default_rng(seed)
    filas = []
    for _, ev in su_si.iterrows():
        fechas = cand.get(int(ev["Codigo_pluvio"]))
        if fechas is None:
            continue
        veto = excl_su.get(ev["su_uid"])
        if veto is not None and len(veto):
            fechas = fechas[~pd.Index(fechas.normalize().date).isin(veto)]
        if len(fechas) < CFG["RATIO_NO_SI"]:
            continue
        for t in rng.choice(fechas.values, CFG["RATIO_NO_SI"], replace=False):
            fila = ev.drop(labels=["geometry"], errors="ignore").to_dict()
            fila[CFG["COL_FECHA"]] = pd.Timestamp(t)
            fila["si_no"] = 0
            filas.append(fila)
    return merge_oni(pd.DataFrame(filas))


# --------------------------------------------------------------------
# Barrido
# --------------------------------------------------------------------
registros = []

for mc in range(N_MC_SWEEP):
    seed = CFG["BASE_SEED"] + mc
    logging.info("=== sweep MC %d/%d | seed=%d ===", mc + 1, N_MC_SWEEP, seed)

    df_no = sample_absences(seed)
    df = pd.concat([su_si.drop(columns="geometry", errors="ignore"), df_no],
                   ignore_index=True)
    df = df.dropna(subset=["ENSO_code"]).reset_index(drop=True)

    # perfiles de lluvia: se calculan UNA vez por corrida
    M = profile_for(df)
    y = df["si_no"].astype(int).to_numpy()
    enso = df["ENSO_code"].astype(float).to_numpy()

    # filtro húmedo simétrico con las columnas del perfil (1d y 30d)
    wet = (M[:, 0] > CFG["UMBRAL_1D"]) & (M[:, 29] > CFG["UMBRAL_30D"])
    keep = wet & ~np.isnan(M[:, 0]) & ~np.isnan(M[:, 29])
    M, y, enso = M[keep], y[keep], enso[keep]
    logging.info("  obs tras filtro húmedo: %d (SI=%d)", len(y), int(y.sum()))

    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=CFG["TEST_SIZE"],
        stratify=y, random_state=seed)

    for t in SHORT:
        for p in LONG:
            if t + p > MAX_DAYS:
                continue
            STR = M[:, t - 1]
            LTR = M[:, t + p - 1] - M[:, t - 1]
            ok = ~np.isnan(STR) & ~np.isnan(LTR)
            X = np.column_stack([STR, LTR, enso])

            tr = np.intersect1d(idx_tr, np.where(ok)[0])
            te = np.intersect1d(idx_te, np.where(ok)[0])
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            try:
                gam = LogisticGAM(TERMS, lam=10.0).fit(X[tr], y[tr])
                auc = roc_auc_score(y[te], gam.predict_proba(X[te]))
                registros.append({"mc_id": mc, "T_days": t, "P_days": p,
                                  "auc": float(auc), "n": int(ok.sum())})
            except Exception as e:
                logging.debug("T=%d P=%d falló: %s", t, p, e)

# --------------------------------------------------------------------
# Agregación
# --------------------------------------------------------------------
raw = pd.DataFrame(registros)
out = CFG["OUT_DIR"]
os.makedirs(out, exist_ok=True)
raw.to_csv(os.path.join(out, "window_search_raw.csv"), index=False)

summary = (raw.groupby(["T_days", "P_days"])["auc"]
           .agg(["median", "mean", "std", "count"]).reset_index()
           .sort_values("median", ascending=False))
summary.to_csv(os.path.join(out, "window_search_summary.csv"), index=False)

logging.info("TOP 10 combinaciones por MEDIANA de AUROC:\n%s",
             summary.head(10).to_string(index=False))
logging.info("=== %s LISTO ===", TAG)
