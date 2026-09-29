"""sens_q_03_resumen.py
=====================

Resumen de la prueba de sensibilidad del dominio (q = 5, 10, 15, 20 %).

Lee las salidas de gam_umbrales_cap6.py de cada q (out_qXX/) y calcula lo mismo
que el notebook umbrales_gam_cap6.ipynb, con sus mismas funciones, de modo que
la fila q = 10 reproduce el capítulo:

  1. dominio: % de SU excluidas (total y por zona) y SU elegibles como ausencia;
  2. desempeño en la parte de prueba: mediana y P5-P95 de las 100 corridas;
  3. cortes P*95, P*85, P*70: mediana y P5-P95 de las corridas;
  4. contrastes de factores (log-odds): modelo final y mediana de las corridas;
  5. STR1 (mm) necesario para cada clase, para cada zona, fase ENSO y LTR30 =
     P25/P50/P75 de las presencias (tipo de SU más frecuente):
        final = modelo final + cortes medianos (lo que dibuja la Fig. de isolíneas)
        runs  = cada corrida con su modelo y su corte -> mediana y P5-P95;
  6. cambio frente a q = 10 %: |Δ| de la mediana, razón Δ / ancho P5-P95 y
     IC 95 % bootstrap de la diferencia de medianas (Very high, fase neutra, P50);
  7. la tabla LaTeX tab:res-gam-sens.

POR QUÉ SE COMPARA LA MEDIANA DE LAS CORRIDAS Y NO EL MODELO FINAL
    El modelo final es UNA corrida (la de AUROC más cercana a la mediana). Con
    otro q se elige otra corrida, y sus umbrales cambian por el sorteo aunque q
    no influya. La mediana de las 100 corridas aísla el efecto de q.

Uso:
    python sens_q_03_resumen.py [carpeta_sens]     (por defecto Umbrales/sensibilidad_q)
Variables opcionales: CAP6_CH6_DIR, CAP6_PATH_SU_TODAS.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

CH6_DIR = Path(os.environ.get("CAP6_CH6_DIR", "/home/oisanchezp/Thesis/src/06_Chapter"))
SENS = Path(sys.argv[1]) if len(sys.argv) > 1 else CH6_DIR / "Umbrales" / "sensibilidad_q"
PATH_SU_TODAS = Path(os.environ.get("CAP6_PATH_SU_TODAS",
                                    CH6_DIR / "salidas" / "su_resultados_ZINB_BYM2.gpkg"))
QS = ["05", "10", "15", "20"]
Q_REF = 10
TAG = "gam_cap6"
TXT = {"zona": str, "tipo_su": str, "level": str}
ZONA = {"1": "Oriente", "2": "Norte", "3": "Sur", "4": "SAP", "5": "Occidente"}
ENSO_ORDER = ["La Niña", "Neutro", "El Niño"]
LEVEL_KEY = {"Moderate": "P_TPR95", "High": "P_TPR85", "Very high": "P_TPR70"}
STR_FINE = np.arange(0.0, 150.0 + 1e-9, 0.25)        # igual que el notebook
N_BOOT, SEED = 4000, 42


# ------------------------------------------------ funciones del notebook
def componentes(e: pd.DataFrame) -> dict:
    g = lambda t: e[e["term"] == t]
    return {"b0": float(g("intercept")["effect"].iloc[0]),
            "STR": (g("STR")["x"].to_numpy(float), g("STR")["effect"].to_numpy(float)),
            "LTR": (g("LTR")["x"].to_numpy(float), g("LTR")["effect"].to_numpy(float)),
            "ENSO": dict(zip(g("ENSO")["level"], g("ENSO")["effect"])),
            "zona": dict(zip(g("zona")["level"], g("zona")["effect"])),
            "tipo_su": dict(zip(g("tipo_su")["level"], g("tipo_su")["effect"]))}


def logit_de(c, STR, LTR, enso, zona, tipo):
    return (c["b0"] + np.interp(STR, *c["STR"]) + np.interp(LTR, *c["LTR"])
            + c["ENSO"][enso] + c["zona"][zona] + c["tipo_su"][tipo])


def str_needed(c, enso, zona, tipo, ltr, pstar) -> float:
    """Primer STR (mm) con p >= pstar a un LTR dado; interpolado. inf si no se alcanza."""
    p = expit(logit_de(c, STR_FINE, np.full_like(STR_FINE, ltr), enso, zona, tipo))
    k = np.nonzero(p >= pstar)[0]
    if len(k) == 0:
        return np.inf
    k = int(k[0])
    if k == 0:
        return 0.0
    return float(STR_FINE[k - 1] + (pstar - p[k - 1]) * (STR_FINE[k] - STR_FINE[k - 1])
                 / (p[k] - p[k - 1]))


# ------------------------------------------------ lectura
def leer(q: str) -> dict:
    d = SENS / f"out_q{q}"
    P = lambda n: d / f"{TAG}_{n}"
    if not P("resumen.json").is_file():
        raise FileNotFoundError(f"Faltan las salidas de q = {q}: {d}")
    o = {"res": json.load(open(P("resumen.json"), encoding="utf-8")),
         "met": pd.read_csv(P("mc_metrics.csv")),
         "thr": pd.read_csv(P("mc_thresholds.csv")),
         "eff": pd.read_csv(P("mc_effects.csv"), dtype=TXT),
         "fin": pd.read_csv(P("efectos_final.csv"), dtype=TXT),
         "ev": pd.read_csv(P("eventos.csv"), dtype=TXT),
         "aus": pd.read_csv(P("mc_ausencias.csv"), usecols=["zona"], dtype={"zona": str})}
    o["cut_run"] = o["thr"].pivot(index="run_id", columns="thr_level", values="threshold")
    o["pstar"] = o["thr"].groupby("thr_level")["threshold"].median()
    o["cf"] = componentes(o["fin"])
    o["cr"] = {r: componentes(g) for r, g in o["eff"].groupby("run_id")}
    return o


def nivel(s: pd.Series) -> pd.Series:
    return s.astype(float).astype(int).astype(str)


# ------------------------------------------------ 1) dominio
def tabla_dominio() -> pd.DataFrame:
    import pyogrio
    todas = pyogrio.read_dataframe(PATH_SU_TODAS, layer="susceptibilidad",
                                   read_geometry=False, columns=["su_id", "zona"])
    todas["zona"] = nivel(todas["zona"])
    n_tot = todas.groupby("zona").size()
    filas = {}
    for q in QS:
        d = pyogrio.read_dataframe(SENS / f"dominio_q{q}.gpkg", layer="susceptibles",
                                   read_geometry=False, columns=["su_id", "zona"])
        d["zona"] = nivel(d["zona"])
        f = {"SU_retenidas": len(d), "excluidas_%": 100 * (1 - len(d) / len(todas))}
        for z, n in (1 - d.groupby("zona").size() / n_tot).items():
            f[f"excluidas_{ZONA[z]}_%"] = 100 * n
        filas[int(q)] = f
    return pd.DataFrame(filas).T


# ------------------------------------------------ 2-4) desempeño, cortes, contrastes
def tabla_modelo(O: dict) -> pd.DataFrame:
    filas = []
    contrastes = {"Oriente-Sur": lambda c: c["zona"]["1"] - c["zona"]["3"],
                  "LaNina-Neutro": lambda c: c["ENSO"]["La Niña"] - c["ENSO"]["Neutro"],
                  "tipo2-tipo1": lambda c: c["tipo_su"]["2"] - c["tipo_su"]["1"]}
    for q, o in O.items():
        met = o["met"]
        f = {"q": int(q), "su_elegibles_ausencia": o["res"]["su_pool"],
             "presencias": o["res"]["presencias"], "corridas": int(met["run_id"].nunique()),
             "run_final": o["res"]["modelo_final"]["run_star"]}
        for m in ("auc", "ap", "brier", "hk"):
            f[f"{m}_med"], f[f"{m}_p05"], f[f"{m}_p95"] = (met[m].median(), met[m].quantile(.05),
                                                        met[m].quantile(.95))
        for k in LEVEL_KEY.values():
            f[k], f[f"{k}_p05"], f[f"{k}_p95"] = (o["pstar"][k], o["cut_run"][k].quantile(.05),
                                                  o["cut_run"][k].quantile(.95))
        for n, fn in contrastes.items():
            v = np.array([fn(c) for c in o["cr"].values()])
            f[f"{n}_final"], f[f"{n}_runs_med"] = fn(o["cf"]), np.median(v)
            f[f"{n}_runs_p05"], f[f"{n}_runs_p95"] = np.percentile(v, [5, 95])
        aus = nivel(o["aus"]["zona"]).map(ZONA).value_counts(normalize=True)
        for z in ZONA.values():
            f[f"ausencias_{z}_%"] = 100 * aus.get(z, 0.0)
        filas.append(f)
    return pd.DataFrame(filas).set_index("q")


# ------------------------------------------------ 5) umbrales en mm
def tabla_mm(O: dict) -> tuple:
    filas, por_corrida = [], {}
    for q, o in O.items():
        ev = o["ev"]
        ltr_ref = dict(zip(["dry (P25)", "average (P50)", "wet (P75)"],
                           np.nanpercentile(ev["LTR"].astype(float), [25, 50, 75])))
        tipo = ev["tipo_su"].astype(str).mode().iloc[0]
        for z in sorted(o["cf"]["zona"], key=float):
            for e in ENSO_ORDER:
                for lab, ltr in ltr_ref.items():
                    u = {"q": int(q), "zona": z, "zone": ZONA[z], "ENSO": e, "tipo_su": tipo,
                         "antecedent": lab, "LTR_mm": ltr}
                    for c, k in LEVEL_KEY.items():
                        runs = np.array([str_needed(o["cr"][r], e, z, tipo, ltr,
                                                    o["cut_run"].loc[r, k]) for r in o["cr"]])
                        u[f"{c}_final"] = str_needed(o["cf"], e, z, tipo, ltr, o["pstar"][k])
                        u[f"{c}_runs_med"] = float(np.median(runs))
                        u[f"{c}_runs_p05"], u[f"{c}_runs_p95"] = np.percentile(runs, [5, 95])
                        if c == "Very high" and e == "Neutro" and lab == "average (P50)":
                            por_corrida[(int(q), ZONA[z])] = runs
                    filas.append(u)
    return pd.DataFrame(filas), por_corrida


# ------------------------------------------------ 6) cambios frente a q = 10
def cambios(u: pd.DataFrame, por_corrida: dict) -> pd.DataFrame:
    idx = ["zone", "ENSO", "antecedent"]
    ref = u[u.q == Q_REF].set_index(idx)
    print("\n== Cambio de la mediana de las corridas frente a q = 10 % (todas las zonas, fases y LTR)")
    for c in LEVEL_KEY:
        s = u[u.q != Q_REF].set_index(idx)
        d = (s[f"{c}_runs_med"] - ref[f"{c}_runs_med"].reindex(s.index)).abs()
        w = (ref[f"{c}_runs_p95"] - ref[f"{c}_runs_p05"]).reindex(s.index)
        print(f"  {c:9s}: |Δ| mediana {d.median():.1f} mm, máx {d.max():.1f} mm "
              f"({s.index[d.argmax()]}, q = {int(s['q'].iloc[d.argmax()])}) | "
              f"Δ/ancho P5-P95 máx {(d / w).max():.2f} | ancho P5-P95 (q=10) "
              f"{ref[f'{c}_runs_p95'].sub(ref[f'{c}_runs_p05']).min():.1f}-"
              f"{ref[f'{c}_runs_p95'].sub(ref[f'{c}_runs_p05']).max():.1f} mm")
    rng = np.random.default_rng(SEED)
    filas = []
    for z in ZONA.values():
        r0 = por_corrida[(Q_REF, z)]
        for q in (int(x) for x in QS if int(x) != Q_REF):
            x = por_corrida[(q, z)]
            bs = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(r0, len(r0)))
                  for _ in range(N_BOOT)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            filas.append({"zone": z, "q": q, "dif_mm": np.median(x) - np.median(r0),
                          "ic95_lo": lo, "ic95_hi": hi, "fuera_del_ruido": not (lo <= 0 <= hi)})
    b = pd.DataFrame(filas)
    print("\n== Very high, fase neutra, LTR = P50: diferencia de medianas frente a q = 10 % "
          "(IC 95 % bootstrap de las corridas)")
    print(b.round(1).to_string(index=False))
    return b


# ------------------------------------------------ 7) tabla LaTeX
def latex(dom: pd.DataFrame, t: pd.DataFrame, u: pd.DataFrame) -> str:
    qs = [int(q) for q in QS]
    v = u[(u.ENSO == "Neutro") & (u.antecedent == "average (P50)")].set_index(["zone", "q"])
    ltr50 = float(u.loc[u.antecedent == "average (P50)", "LTR_mm"].iloc[0])
    miles = lambda n: f"{int(round(n)):,}".replace(",", "{,}")
    rng = lambda a, b, fmt: f"{fmt(a)}--{fmt(b)}"

    def fila(nombre, vals, rango="--"):
        return f"{nombre} & " + " & ".join(vals) + f" & {rango} \\\\"

    L = [r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
         r"\begin{tabular}{lccccc}", r"\hline",
         r" & \multicolumn{4}{c}{\textbf{Percentile $q$}} & \textbf{Range over} \\",
         r"\cline{2-5}",
         r" & \textbf{5~\%} & \textbf{10~\% (ref.)} & \textbf{15~\%} & \textbf{20~\%} "
         r"& \textbf{the runs ($q = 10~\%$)} \\", r"\hline",
         r"\multicolumn{6}{l}{\emph{Analysis domain}} \\",
         fila("Excluded slope units (\\%)", [f"{dom.loc[q, 'excluidas_%']:.1f}" for q in qs]),
         fila("Units eligible as non-landslides",
              [miles(t.loc[q, "su_elegibles_ausencia"]) for q in qs]),
         fila("Landslide observations", [f"{int(t.loc[q, 'presencias'])}" for q in qs]),
         r"\multicolumn{6}{l}{\emph{Performance on the test part}} \\"]
    for k, nombre in [("auc", "AUROC"), ("ap", "Average precision"), ("brier", "Brier score")]:
        L.append(fila(nombre, [f"{t.loc[q, f'{k}_med']:.3f}" for q in qs],
                      rng(t.loc[Q_REF, f"{k}_p05"], t.loc[Q_REF, f"{k}_p95"], lambda x: f"{x:.3f}")))
    L.append(r"\multicolumn{6}{l}{\emph{Probability cut-offs}} \\")
    for k, nombre in [("P_TPR95", r"$P^{*}_{95}$"), ("P_TPR85", r"$P^{*}_{85}$"),
                      ("P_TPR70", r"$P^{*}_{70}$")]:
        L.append(fila(nombre, [f"{t.loc[q, k]:.3f}" for q in qs],
                      rng(t.loc[Q_REF, f"{k}_p05"], t.loc[Q_REF, f"{k}_p95"], lambda x: f"{x:.3f}")))
    L.append(r"\multicolumn{6}{l}{\emph{$\mathrm{STR}_{1}$ needed to reach Very high (mm)}} \\")
    for z in ["Oriente", "Norte", "Sur", "SAP", "Occidente"]:
        L.append(fila(z, [f"{v.loc[(z, q), 'Very high_runs_med']:.0f}" for q in qs],
                      rng(v.loc[(z, Q_REF), "Very high_runs_p05"], v.loc[(z, Q_REF), "Very high_runs_p95"],
                          lambda x: f"{x:.0f}")))
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Effect of the percentile $q$ that sets the analysis domain on the rainfall "
          r"model. Values are medians over the 100 Monte Carlo runs of each $q$; the last column "
          r"gives the 5th--95th percentile over the runs of the reference, $q = 10~\%$. The "
          r"thresholds are for slope units of type SU~1 in the neutral ENSO phase, with "
          rf"$\mathrm{{LTR}}_{{30}} = {ltr50:.0f}$~mm, the median of the landslide observations. "
          r"Each run uses its own model and cut-off, so these medians differ slightly from the "
          r"final model drawn in Fig.~\ref{fig:res-gam-isolines}.}",
          r"\label{tab:res-gam-sens}", r"\end{table}"]
    return "\n".join(L)


def main() -> None:
    O = {q: leer(q) for q in QS}
    dom = tabla_dominio()
    t = tabla_modelo(O)
    u, por_corrida = tabla_mm(O)
    b = cambios(u, por_corrida)
    tex = latex(dom, t, u)

    dom.to_csv(SENS / "tabla_sens_q_dominio.csv")
    t.to_csv(SENS / "tabla_sens_q_modelo.csv")
    u.to_csv(SENS / "tabla_sens_q_umbrales_mm.csv", index=False)
    b.to_csv(SENS / "tabla_sens_q_bootstrap_very_high.csv", index=False)
    (SENS / "tab_res_gam_sens.tex").write_text(tex + "\n", encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n== Dominio\n" + dom.round(1).to_string())
    print("\n== Modelo\n" + t.T.round(4).to_string())
    v = u[(u.ENSO == "Neutro") & (u.antecedent == "average (P50)")]
    for c in LEVEL_KEY:
        print(f"\n== {c}: STR1 (mm), fase neutra, LTR30 = P50 — mediana de las corridas")
        print(v.pivot(index="zone", columns="q", values=f"{c}_runs_med").round(1).to_string())
    print("\n== LaTeX\n" + tex)


if __name__ == "__main__":
    main()