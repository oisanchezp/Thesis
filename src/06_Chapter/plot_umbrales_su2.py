"""
plot_umbrales_su2.py
====================

Genera figuras de umbrales operacionales sobre el plano (lluvia_30d × lluvia_1d)
usando las salidas de `gam_umbrales_su2_ensayo.py`.

NIVELES DE ALERTA (tres umbrales mostrados)
-------------------------------------------
Se muestran TRES isolíneas (la de TPR=85% se omite porque queda muy
solapada con OPT en este modelo).

    Crítica (TPR=70%)     ->  línea AZUL continua    (umbral más exigente, P* alto)
    Óptimo (Youden)       ->  línea ROJA DISCONTINUA (referencia matemática)
    Vigilancia (TPR=95%)  ->  línea NEGRA continua   (umbral más permisivo, P* bajo)

CONVENCIÓN DE NOMBRES (consultar con el lector)
-----------------------------------------------
- "Vigilancia" = modo MÁS SENSIBLE: detecta el 95% de eventos. Probabilidad
  de corte baja => la alerta se dispara seguido. Operacionalmente: vigilar
  activamente cuando haya lluvia significativa.
- "Crítica" = modo MÁS RESTRICTIVO: detecta el 70% de eventos. Probabilidad
  de corte alta => solo se dispara con condiciones de muy alta probabilidad
  de deslizamiento.

Nota: el archivo del entrenamiento sigue calculando y guardando los cuatro
umbrales (P_TPR70, P_OPT, P_TPR85, P_TPR95) — aquí solo se omite P_TPR85
de la VISUALIZACIÓN. Para volver a mostrarlo, agregar su entrada en
ALERT_LEVELS abajo.

OPT como referencia, no como umbral operacional
-----------------------------------------------
OPT (Youden, max TPR-FPR) asume costos simétricos FP/FN, lo cual no aplica
a un sistema de alerta temprana donde el costo de no detectar >> falsa
alarma. Se grafica como referencia matemática.

CONVENCIÓN DE EJES (Caine 1980, Guzzetti et al. 2008, Steger et al. 2024)
    eje X : lluvia antecedente / acumulada (lluvia_30d, mm)
    eje Y : lluvia detonante / corto plazo (lluvia_1d, mm)

ESTILO DE PUNTOS
    Eventos (SI):                círculos ROJOS con borde blanco
    No-eventos run_star:         cruces AZULES opacas
    No-eventos TODAS las corridas (opcional): cruces NEGRAS translúcidas

ZONAS DE LLUVIA (SIATA)
    1 -> Oriente, 2 -> Norte, 3 -> Sur, 4 -> SAP, 5 -> Occidente

Lee:
    OUT_DIR/gam_su2_grid_probs.parquet
    OUT_DIR/gam_su2_thresholds_mc.csv
    OUT_DIR/gam_su2_dataset_run_star.parquet
    OUT_DIR/gam_su2_no_events_all_runs.parquet  [opcional]
    OUT_DIR/gam_su2_roc_curves_mc.parquet       [requerido para fig ROC]
    OUT_DIR/gam_su2_package.json                [requerido para P* operacionales]

Produce:
    - fig_separabilidad_global.png
    - fig_umbrales_por_zona_<ENSO>.png
    - fig_umbrales_por_enso_zona<i>_<nombre>.png
    - fig_thresholds_mc_boxplot.png
    - fig_roc_su2.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score

# ============================================================
# CONFIG
# ============================================================

OUT_DIR = "/home/oisanchezp/Thesis/src/06_Chapter/Umbrales/"

GRID_AGG     = os.path.join(OUT_DIR, "gam_su2_grid_probs.parquet")
THR_CSV      = os.path.join(OUT_DIR, "gam_su2_thresholds_mc.csv")
DATASET_STAR = os.path.join(OUT_DIR, "gam_su2_dataset_run_star.parquet")
NO_ALL_PQ    = os.path.join(OUT_DIR, "gam_su2_no_events_all_runs.parquet")
ROC_PQ       = os.path.join(OUT_DIR, "gam_su2_roc_curves_mc.parquet")
PKG_JSON     = os.path.join(OUT_DIR, "gam_su2_package.json")

# Niveles del contourf (fondo coloreado)
PROB_LEVELS_FILL = np.linspace(0, 1, 11)

# Definición operacional de los TRES niveles a graficar.
# Convención de nombres (invertida respecto a "más TPR = más crítico"):
#   - "Crítica"    = umbral más exigente (TPR bajo, P* alto). Se dispara poco.
#   - "Óptimo"     = referencia matemática (Youden).
#   - "Vigilancia" = umbral más permisivo (TPR alto, P* bajo). Se dispara seguido.
#
# Orden de la lista = orden visual en la leyenda y en el boxplot.
# P_TPR85 se omite deliberadamente porque se solapa con P_OPT.
ALERT_LEVELS = [
    {"key": "P_TPR70", "color": "#1f77b4", "tpr": 0.70, "name": "Crítica",
     "linestyle": "-",  "is_operational": True},
    {"key": "P_OPT",   "color": "#d62728", "tpr": None, "name": "Óptimo (Youden)",
     "linestyle": "--", "is_operational": False},
    {"key": "P_TPR95", "color": "#000000", "tpr": 0.95, "name": "Vigilancia",
     "linestyle": "-",  "is_operational": True},
]

# Fallback si no se encuentra package.json
PROB_TARGETS_FALLBACK = {"P_TPR70": 0.70, "P_OPT": 0.50, "P_TPR95": 0.95}

# Estilos de scatter
STYLE_SI = dict(s=42, c="#d62728", marker="o", edgecolor="white",
                lw=1.0, zorder=12, alpha=0.95)
STYLE_NO_STAR = dict(s=22, c="#2787F5", marker="x", lw=1.0,
                     zorder=10, alpha=0.85)
STYLE_NO_ALL  = dict(s=8,  c="#000000", marker="x", lw=0.5,
                     zorder=6, alpha=0.5)

ZONA_NAMES = {
    "1": "Oriente",
    "2": "Norte",
    "3": "Sur",
    "4": "SAP",
    "5": "Occidente",
}

ENSO_NORMALIZE = {"Neutral": "Neutro"}


# ============================================================
# CARGA DE UMBRALES OPERACIONALES
# ============================================================

def load_operational_thresholds(pkg_path: str = PKG_JSON):
    """
    Lee los P* operacionales medianos desde package.json.

    Returns
    -------
    list of dict, cada uno con: key, color, tpr (nominal o None), name,
        linestyle, is_operational, p_star, p_star_low, p_star_high,
        tpr_actual, fpr_actual (estos dos últimos relevantes para OPT).
    """
    if not os.path.exists(pkg_path):
        warnings.warn(
            f"No se encontró {pkg_path}; usando fallback. "
            "Las isolíneas NO corresponderán a P* operacionales.",
            UserWarning,
        )
        out = []
        for lvl in ALERT_LEVELS:
            p = PROB_TARGETS_FALLBACK.get(lvl["key"], 0.5)
            out.append({**lvl, "p_star": p, "p_star_low": p, "p_star_high": p,
                        "tpr_actual": lvl.get("tpr") or 0.0, "fpr_actual": 0.0})
        return out

    with open(pkg_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)
    thr_dict = pkg["summary_mc"]["thresholds"]

    out = []
    for lvl in ALERT_LEVELS:
        info = thr_dict.get(lvl["key"])
        if info is None:
            warnings.warn(f"Clave {lvl['key']} no encontrada en package.json.",
                           UserWarning)
            continue
        out.append({
            **lvl,
            "p_star":      float(info["median"]),
            "p_star_low":  float(info["p05"]),
            "p_star_high": float(info["p95"]),
            "tpr_actual":  float(info.get("tpr_median", lvl.get("tpr") or 0.0)),
            "fpr_actual":  float(info.get("fpr_median", 0.0)),
        })
    return out


def _alert_label(lvl: dict) -> str:
    """Etiqueta para la isolínea."""
    if lvl["key"] == "P_OPT":
        return (f"{lvl['name']} (P*={lvl['p_star']:.2f}, "
                f"TPR={lvl['tpr_actual']:.0%})")
    return f"{lvl['name']} (TPR={int(lvl['tpr']*100)}%, P*={lvl['p_star']:.2f})"


# ============================================================
# UTILIDADES
# ============================================================

def zona_label(z) -> str:
    if isinstance(z, (int, float)) and not pd.isna(z):
        key = str(int(float(z)))
    else:
        key = str(z).strip()
    if key.endswith(".0"):
        key = key[:-2]
    return ZONA_NAMES.get(key, f"Zona {key}")


def _grid_pivot(grid_zona_enso: pd.DataFrame, value_col: str):
    pivot = grid_zona_enso.pivot_table(
        index="lluvia_1d", columns="lluvia_30d",
        values=value_col, aggfunc="mean",
    )
    Y = pivot.index.values
    X = pivot.columns.values
    Z = pivot.values
    XX, YY = np.meshgrid(X, Y, indexing="xy")
    return XX, YY, Z


def _normalize_enso(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace(ENSO_NORMALIZE)


def _filter_eventos(df: pd.DataFrame, zona, enso: str,
                    label: int = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mask = (
        (df["zona"].astype(str) == str(zona)) &
        (_normalize_enso(df["ENSO"]) == enso)
    )
    if label is not None and "si_no" in df.columns:
        mask = mask & (df["si_no"] == label)
    return df.loc[mask].copy()


def _draw_panel(ax, sub_grid: pd.DataFrame,
                eventos_si: pd.DataFrame, eventos_no_star: pd.DataFrame,
                eventos_no_all: pd.DataFrame,
                alert_levels: list,
                title: str):
    """Contornos + isolíneas en P* operacionales (incluye OPT) + scatter."""
    if sub_grid.empty:
        ax.set_title(f"{title} (sin datos)")
        ax.axis("off")
        return None

    XX, YY, Z = _grid_pivot(sub_grid, "prob_median")

    cf = ax.contourf(XX, YY, Z, levels=PROB_LEVELS_FILL,
                     cmap="Spectral_r", alpha=0.65)

    # Las isolíneas deben dibujarse UNA POR UNA porque matplotlib.contour
    # no permite linestyles distintos en una sola llamada con `levels=[...]`.
    # Esto además garantiza que cada línea tenga su color y estilo propios.
    for lvl in alert_levels:
        p = lvl["p_star"]
        cs = ax.contour(XX, YY, Z, levels=[p],
                         colors=[lvl["color"]],
                         linestyles=[lvl["linestyle"]],
                         linewidths=2.2)
        if lvl["key"] == "P_OPT":
            fmt = {p: f"OPT P*={p:.2f}"}
        else:
            fmt = {p: f"{lvl['name']} P*={p:.2f}"}
        ax.clabel(cs, fmt=fmt, inline=True, fontsize=8)

    n_no_all = 0
    if eventos_no_all is not None and not eventos_no_all.empty:
        ax.scatter(eventos_no_all["30d"], eventos_no_all["1d"], **STYLE_NO_ALL)
        n_no_all = len(eventos_no_all)

    n_no_star = 0
    if eventos_no_star is not None and not eventos_no_star.empty:
        ax.scatter(eventos_no_star["30d"], eventos_no_star["1d"], **STYLE_NO_STAR)
        n_no_star = len(eventos_no_star)

    n_si = 0
    if eventos_si is not None and not eventos_si.empty:
        ax.scatter(eventos_si["30d"], eventos_si["1d"], **STYLE_SI)
        n_si = len(eventos_si)

    if n_no_all > 0:
        subtitle = f"SI={n_si} | NO_star={n_no_star} | NO_all={n_no_all}"
    else:
        subtitle = f"SI={n_si} | NO={n_no_star}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("Lluvia 30 días (mm)")
    ax.set_ylabel("Lluvia 1 día (mm)")
    ax.set_xlim(float(np.nanmin(XX)), float(np.nanmax(XX)))
    ax.set_ylim(float(np.nanmin(YY)), float(np.nanmax(YY)))
    ax.grid(True, alpha=0.3)
    return cf


def _legend_handles(alert_levels: list, has_no_all: bool):
    """Leyenda: cuatro entradas de isolíneas + entradas de puntos."""
    handles = []
    # Isolíneas operacionales (líneas con su linestyle propio)
    for lvl in alert_levels:
        handles.append(
            Line2D([0], [0], color=lvl["color"], lw=2.4,
                   linestyle=lvl["linestyle"],
                   label=_alert_label(lvl))
        )
    # Separador visual
    handles.append(Line2D([0], [0], color="none", label=" "))
    # Puntos
    handles.append(
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="#d62728", markeredgecolor="white",
               markeredgewidth=1.0, label="Deslizamientos (SI)")
    )
    handles.append(
        Line2D([0], [0], marker="x", linestyle="", markersize=9,
               markeredgecolor="#2787F5", markeredgewidth=1.6,
               label="No-eventos run_star")
    )
    if has_no_all:
        handles.append(
            Line2D([0], [0], marker="x", linestyle="", markersize=8,
                   markeredgecolor="black", markeredgewidth=1.0,
                   alpha=0.5, label="No-eventos todas las corridas MC")
        )
    return handles


# ============================================================
# FIGURA: PANELES POR ZONA (ENSO fijo)
# ============================================================

def plot_panel_por_zona(grid_agg: pd.DataFrame, enso_fijo: str,
                         alert_levels: list,
                         eventos_df: pd.DataFrame = None,
                         no_all_df: pd.DataFrame = None,
                         outdir: str = OUT_DIR) -> str:
    zonas = sorted(grid_agg["zona"].unique(), key=lambda v: str(v))
    n = len(zonas)
    ncols_panels = min(3, n)
    nrows = int(np.ceil(n / ncols_panels))
    has_spare = (nrows * ncols_panels) > n

    fig = plt.figure(figsize=(5.0 * ncols_panels + 1.4, 4.6 * nrows + 0.6))
    width_ratios = [1.0] * ncols_panels + [0.06]
    gs = gridspec.GridSpec(
        nrows=nrows, ncols=ncols_panels + 1,
        width_ratios=width_ratios,
        figure=fig, wspace=0.28, hspace=0.42,
    )

    axes = []
    for i, z in enumerate(zonas):
        r = i // ncols_panels
        c = i % ncols_panels
        axes.append(fig.add_subplot(gs[r, c]))
    cax = fig.add_subplot(gs[:, -1])

    cf = None
    for ax, z in zip(axes, zonas):
        sub = grid_agg[(grid_agg["zona"] == z) & (grid_agg["ENSO"] == enso_fijo)]
        ev_si      = _filter_eventos(eventos_df, z, enso_fijo, label=1)
        ev_no_star = _filter_eventos(eventos_df, z, enso_fijo, label=0)
        ev_no_all  = _filter_eventos(no_all_df,  z, enso_fijo)
        cf_panel = _draw_panel(ax, sub, ev_si, ev_no_star, ev_no_all,
                                alert_levels=alert_levels,
                                title=f"Zona {z} — {zona_label(z)}")
        if cf_panel is not None:
            cf = cf_panel

    if cf is not None:
        fig.colorbar(cf, cax=cax,
                     label="Probabilidad de deslizamiento (mediana MC)")
    else:
        cax.axis("off")

    has_no_all = (no_all_df is not None and not no_all_df.empty)
    handles = _legend_handles(alert_levels, has_no_all)
    if has_spare:
        r_free = (n - 1) // ncols_panels
        c_free = (n) % ncols_panels
        if c_free == 0:
            r_free += 1
            c_free = 0
        ax_leg = fig.add_subplot(gs[r_free, c_free])
        ax_leg.axis("off")
        ax_leg.legend(handles=handles, loc="center", frameon=False,
                      fontsize=10, title="Niveles de alerta y muestras",
                      title_fontsize=10)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"Umbrales operacionales | su_type=2 | ENSO: {enso_fijo}",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.92, bottom=0.10 if not has_spare else 0.06,
                         left=0.07, right=0.93)

    out_path = os.path.join(
        outdir, f"fig_umbrales_por_zona_{enso_fijo.replace(' ', '_')}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# FIGURA: PANELES POR ENSO (zona fija)
# ============================================================

def plot_panel_por_enso(grid_agg: pd.DataFrame, zona_fija,
                         alert_levels: list,
                         eventos_df: pd.DataFrame = None,
                         no_all_df: pd.DataFrame = None,
                         outdir: str = OUT_DIR) -> str:
    ensos = ["La Niña", "Neutro", "El Niño"]
    ncols_panels = 3

    fig = plt.figure(figsize=(5.0 * ncols_panels + 1.4, 5.4))
    width_ratios = [1.0] * ncols_panels + [0.06]
    gs = gridspec.GridSpec(
        nrows=1, ncols=ncols_panels + 1,
        width_ratios=width_ratios,
        figure=fig, wspace=0.28,
    )

    axes = [fig.add_subplot(gs[0, c]) for c in range(ncols_panels)]
    cax  = fig.add_subplot(gs[0, -1])

    cf = None
    for ax, e in zip(axes, ensos):
        sub = grid_agg[(grid_agg["zona"] == zona_fija) & (grid_agg["ENSO"] == e)]
        ev_si      = _filter_eventos(eventos_df, zona_fija, e, label=1)
        ev_no_star = _filter_eventos(eventos_df, zona_fija, e, label=0)
        ev_no_all  = _filter_eventos(no_all_df,  zona_fija, e)
        cf_panel = _draw_panel(ax, sub, ev_si, ev_no_star, ev_no_all,
                                alert_levels=alert_levels,
                                title=f"ENSO: {e}")
        if cf_panel is not None:
            cf = cf_panel

    if cf is not None:
        fig.colorbar(cf, cax=cax,
                     label="Probabilidad de deslizamiento (mediana MC)")
    else:
        cax.axis("off")

    has_no_all = (no_all_df is not None and not no_all_df.empty)
    handles = _legend_handles(alert_levels, has_no_all)
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Umbrales operacionales | su_type=2 | "
        f"Zona {zona_fija} — {zona_label(zona_fija)}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.06, right=0.93)

    safe_label = zona_label(zona_fija).replace(" ", "_")
    out_path = os.path.join(
        outdir, f"fig_umbrales_por_enso_zona{zona_fija}_{safe_label}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# BOXPLOT DE UMBRALES
# ============================================================

def plot_thresholds_boxplot(thr_df: pd.DataFrame,
                             alert_levels: list,
                             outdir: str = OUT_DIR) -> str:
    """Boxplot con tres cajas (Crítica, OPT, Vigilancia)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    levels_order = [lvl["key"]  for lvl in alert_levels]
    labels = []
    for lvl in alert_levels:
        if lvl["key"] == "P_OPT":
            # nombre ya contiene "(Youden)", no agregar otra vez
            labels.append(lvl["name"])
        else:
            labels.append(f"{lvl['name']}\nTPR={int(lvl['tpr']*100)}%")
    box_colors = [lvl["color"] for lvl in alert_levels]
    data = [thr_df[thr_df["thr_level"] == lvl]["threshold"].values
            for lvl in levels_order]

    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.35)
        patch.set_edgecolor(c); patch.set_linewidth(1.4)
    for median_line, c in zip(bp["medians"], box_colors):
        median_line.set_color(c); median_line.set_linewidth(2.0)

    # Anotación de P* mediano al lado de cada caja
    for i, lvl in enumerate(alert_levels, start=1):
        ax.annotate(f"P*={lvl['p_star']:.3f}",
                    xy=(i, lvl["p_star"]),
                    xytext=(28, 0), textcoords="offset points",
                    fontsize=9, va="center", color=lvl["color"],
                    fontweight="bold")

    ax.set_ylabel("Probabilidad umbral P*")
    ax.set_title("Estabilidad MC de los umbrales (su_type=2)")
    ax.grid(True, alpha=0.3)

    # Nota al pie sobre OPT
    fig.text(0.5, 0.005,
             "Nota: OPT (Youden) se reporta como referencia matemática. "
             "Los umbrales operacionales del sistema son Crítica (TPR=70%) y Vigilancia (TPR=95%).",
             ha="center", fontsize=8, style="italic", color="#444444")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(outdir, "fig_thresholds_mc_boxplot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# SEPARABILIDAD GLOBAL
# ============================================================

def plot_separabilidad_global(eventos_df: pd.DataFrame,
                               no_all_df: pd.DataFrame = None,
                               outdir: str = OUT_DIR) -> str:
    """SI vs NO sin estratificar por zona/ENSO."""
    if eventos_df is None or eventos_df.empty:
        return ""
    si      = eventos_df[eventos_df["si_no"] == 1]
    no_star = eventos_df[eventos_df["si_no"] == 0]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    if no_all_df is not None and not no_all_df.empty:
        ax.scatter(no_all_df["30d"], no_all_df["1d"],
                   **STYLE_NO_ALL, label=f"NO todas las corridas (n={len(no_all_df)})")
    ax.scatter(no_star["30d"], no_star["1d"],
               **STYLE_NO_STAR, label=f"NO run_star (n={len(no_star)})")
    ax.scatter(si["30d"], si["1d"],
               **STYLE_SI, label=f"SI (n={len(si)})")

    ax.set_xlabel("Lluvia 30 días (mm)")
    ax.set_ylabel("Lluvia 1 día (mm)")
    ax.set_title("Separabilidad SI vs NO en el plano lluvia 30d–1d")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(outdir, "fig_separabilidad_global.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# FIGURA ROC (NUEVA)
# ============================================================

def plot_roc_with_thresholds(roc_df: pd.DataFrame,
                              alert_levels: list,
                              run_id_star: int,
                              outdir: str = OUT_DIR) -> str:
    """
    Curva ROC de la corrida estrella + banda MC (envolvente de todas las
    corridas) + cuatro puntos operacionales marcados.

    Inspirada en Steger et al. (2024) Fig. 8b: cada punto operacional
    se marca con su color, etiqueta y métricas (P*, TPR, FPR).

    Parameters
    ----------
    roc_df : DataFrame
        Columnas: run_id, y_true, y_prob (vector largo, una fila por
        predicción del test set por corrida).
    alert_levels : list of dict
        Salida de load_operational_thresholds(). Cada dict tiene
        p_star, p_star_low, p_star_high, color, linestyle, name, etc.
    run_id_star : int
        ID de la corrida cuya curva ROC se dibuja como línea principal.
    """
    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # ----- Banda MC: envolvente de todas las curvas ROC -----
    # Interpolamos cada curva en una grilla común de FPR
    fpr_grid = np.linspace(0.0, 1.0, 200)
    tprs_interp = []
    aucs = []
    for rid, sub in roc_df.groupby("run_id"):
        y_true = sub["y_true"].to_numpy(dtype=int)
        y_prob = sub["y_prob"].to_numpy(dtype=float)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        fpr_r, tpr_r, _ = roc_curve(y_true, y_prob)
        # Asegurar inicio en (0,0)
        if fpr_r[0] > 0:
            fpr_r = np.concatenate([[0.0], fpr_r])
            tpr_r = np.concatenate([[0.0], tpr_r])
        tpr_interp = np.interp(fpr_grid, fpr_r, tpr_r)
        tprs_interp.append(tpr_interp)
        aucs.append(roc_auc_score(y_true, y_prob))

    if tprs_interp:
        tprs_mat = np.vstack(tprs_interp)
        tpr_p05 = np.quantile(tprs_mat, 0.05, axis=0)
        tpr_p95 = np.quantile(tprs_mat, 0.95, axis=0)
        ax.fill_between(fpr_grid, tpr_p05, tpr_p95,
                         color="#cccccc", alpha=0.55,
                         label=f"Rango MC p05–p95 (n={len(tprs_interp)} corridas)")

    # ----- Curva ROC de la run_star -----
    star = roc_df[roc_df["run_id"] == run_id_star]
    if star.empty:
        warnings.warn(f"No se encontró run_id={run_id_star} en roc_df.")
        return ""
    y_true_star = star["y_true"].to_numpy(dtype=int)
    y_prob_star = star["y_prob"].to_numpy(dtype=float)
    fpr_s, tpr_s, thr_s = roc_curve(y_true_star, y_prob_star)
    auc_star = roc_auc_score(y_true_star, y_prob_star)
    ax.plot(fpr_s, tpr_s, color="black", lw=2.2,
            label=f"ROC run_star (id={run_id_star}, AUC={auc_star:.3f})")

    # ----- Diagonal de referencia -----
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1.0,
            label="Clasificador aleatorio")

    # ----- Tres puntos operacionales (Crítica, OPT, Vigilancia) -----
    # Cada punto se ubica en (FPR, TPR) del run_star correspondiente al P*.
    point_positions = []
    for lvl in alert_levels:
        idx_closest = int(np.argmin(np.abs(thr_s - lvl["p_star"])))
        point_positions.append({
            "lvl": lvl,
            "fpr": float(fpr_s[idx_closest]),
            "tpr": float(tpr_s[idx_closest]),
        })

    # Dibujar los puntos primero (sin etiquetas)
    for pos in point_positions:
        ax.scatter(pos["fpr"], pos["tpr"], s=130, color=pos["lvl"]["color"],
                   edgecolor="white", linewidth=1.6, zorder=10)

    # ----- Layout de etiquetas: caja única apilada con todos los puntos -----
    # Como los puntos suelen estar muy juntos en la esquina superior-izquierda
    # de una ROC con buen AUC, en lugar de poner una etiqueta por punto (que
    # invariablemente se solapan), se construye una sola caja-leyenda con las
    # métricas, ubicada en la mitad derecha donde hay espacio.

    lines = []
    for pos in point_positions:
        lvl = pos["lvl"]
        # Marcador colorado al inicio de cada línea
        lines.append({
            "color": lvl["color"],
            "text": (f"● {lvl['name']}:  "
                     f"P*={lvl['p_star']:.3f}  |  "
                     f"TPR={pos['tpr']:.1%}  |  FPR={pos['fpr']:.1%}"),
        })

    # Construir caja con texto multicolor línea-a-línea
    box_x = 0.32  # centrar en la mitad-derecha de la ROC
    box_y_top = 0.42
    line_height = 0.045
    box_width = 0.66
    # Fondo de la caja
    box_height = line_height * (len(lines) + 0.6)
    rect_y = box_y_top - box_height + line_height * 0.6
    ax.add_patch(plt.Rectangle(
        (box_x - 0.01, rect_y), box_width, box_height,
        transform=ax.transAxes, fill=True, facecolor="white",
        edgecolor="#444444", lw=0.8, zorder=8, alpha=0.96,
    ))
    # Título
    ax.text(box_x, box_y_top + line_height * 0.3,
            "Umbrales operacionales (run_star)",
            transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color="#222222", zorder=9)
    # Cada nivel en su línea, con su color
    for i, ln in enumerate(lines):
        ax.text(box_x, box_y_top - line_height * (i + 1),
                ln["text"], transform=ax.transAxes,
                fontsize=9, color=ln["color"], fontweight="bold",
                zorder=9, family="monospace")

    # ----- AUC global MC en esquina inferior derecha
    if aucs:
        auc_med = float(np.median(aucs))
        auc_p05 = float(np.quantile(aucs, 0.05))
        auc_p95 = float(np.quantile(aucs, 0.95))
        ax.text(0.97, 0.05,
                f"AUC MC: {auc_med:.3f}\n[p05={auc_p05:.3f}, p95={auc_p95:.3f}]",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#444444", lw=0.8))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curva ROC con umbrales operacionales — su_type=2",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9, framealpha=0.95,
              bbox_to_anchor=(1.0, 0.62))
    ax.set_aspect("equal")

    fig.tight_layout()
    out_path = os.path.join(outdir, "fig_roc_su2.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    grid_agg   = pd.read_parquet(GRID_AGG)
    thr_df     = pd.read_csv(THR_CSV)
    eventos_df = (pd.read_parquet(DATASET_STAR)
                  if os.path.exists(DATASET_STAR) else None)
    no_all_df  = (pd.read_parquet(NO_ALL_PQ)
                  if os.path.exists(NO_ALL_PQ) else None)
    roc_df     = (pd.read_parquet(ROC_PQ)
                  if os.path.exists(ROC_PQ) else None)

    if eventos_df is not None:
        eventos_df["ENSO"] = _normalize_enso(eventos_df["ENSO"])
    if no_all_df is not None:
        no_all_df["ENSO"] = _normalize_enso(no_all_df["ENSO"])
        print(f"[INFO] NO de todas las corridas: {len(no_all_df)} filas, "
              f"{no_all_df['run_id'].nunique()} corridas.")
    else:
        print("[INFO] No se encontró NO_ALL_PQ; se grafican solo NO de run_star.")

    if roc_df is None:
        print("[WARN] No se encontró ROC_PQ; no se generará fig_roc_su2.png.")

    # Cargar P* operacionales una sola vez
    alert_levels = load_operational_thresholds(PKG_JSON)
    print("Umbrales (P* medianos MC):")
    for lvl in alert_levels:
        if lvl["key"] == "P_OPT":
            print(f"  {lvl['name']}: P*={lvl['p_star']:.3f}  "
                  f"[p05={lvl['p_star_low']:.3f}, p95={lvl['p_star_high']:.3f}]  "
                  f"TPR_actual={lvl['tpr_actual']:.2%}, "
                  f"FPR_actual={lvl['fpr_actual']:.2%}")
        else:
            print(f"  {lvl['name']} (TPR={int(lvl['tpr']*100)}%): "
                  f"P*={lvl['p_star']:.3f}  "
                  f"[p05={lvl['p_star_low']:.3f}, p95={lvl['p_star_high']:.3f}]")

    # Métricas del paquete
    run_id_star = None
    if os.path.exists(PKG_JSON):
        with open(PKG_JSON, "r", encoding="utf-8") as fjson:
            pkg = json.load(fjson)
        print("AUC mediana:",   pkg["summary_mc"]["auc_median"])
        print("AP mediana:",    pkg["summary_mc"]["ap_median"])
        print("Brier mediana:", pkg["summary_mc"]["brier_median"])
        run_id_star = pkg["summary_mc"].get("run_id_star")

    p = plot_separabilidad_global(eventos_df, no_all_df=no_all_df)
    if p: print("Guardado:", p)

    for e in sorted(grid_agg["ENSO"].unique()):
        p = plot_panel_por_zona(grid_agg, e, alert_levels=alert_levels,
                                 eventos_df=eventos_df, no_all_df=no_all_df)
        print("Guardado:", p)

    for z in sorted(grid_agg["zona"].unique(), key=lambda v: str(v)):
        p = plot_panel_por_enso(grid_agg, z, alert_levels=alert_levels,
                                 eventos_df=eventos_df, no_all_df=no_all_df)
        print("Guardado:", p)

    p = plot_thresholds_boxplot(thr_df, alert_levels=alert_levels)
    print("Guardado:", p)

    # Figura ROC nueva (solo si tenemos roc_df y run_id_star)
    if roc_df is not None and run_id_star is not None:
        p = plot_roc_with_thresholds(roc_df, alert_levels, run_id_star)
        if p: print("Guardado:", p)


if __name__ == "__main__":
    main()