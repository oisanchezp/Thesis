"""
su_crosscheck.py
================
Cross-analysis between the Slope Unit (SU) types and the landslide inventory.
Chapter 4 -- Aburra Valley thesis.

Complements `eda_slope_unit_types.ipynb`. Produces four deliverables:

    T1  Traceability table of the SU x inventory cross-analysis (pandas + LaTeX)
    F1  Frequency Ratio figure (2 panels: % area vs % landslides, and FR with CI)
    F2  Landslide-density heatmap, SU type x rainfall zone, with margins
    F3  Choropleth maps of landslide activity per SU (counts + EB-smoothed density)

Expected input: a GeoDataFrame in a PROJECTED CRS (metres, e.g. EPSG:3116) with
    su_type      int    1..3   SU morphometric type (GMM cluster)
    n_mm         num           landslides assigned to the unit
    zona_lluvia  num    1..5   rainfall zone (Chapter 3)
    geometry     Polygon

Design notes
------------
* Density is D_t = N_t / A_t (events per km2). The Frequency Ratio is
  FR_t = (N_t/N) / (A_t/A), which is algebraically D_t / D_global. The FR is
  reported because it is dimensionless, comparable across studies, and has a
  natural reference value of 1 (Reichenbach et al., 2018).
* Confidence intervals use a NON-PARAMETRIC BOOTSTRAP over slope units,
  stratified by type. This is preferred over an exact Poisson interval because
  landslide counts cluster within units (overdispersion), which a Poisson
  interval would ignore and therefore report intervals that are too narrow.
* Per-unit densities n_i/A_i are unstable for small units (the small-area
  problem: one landslide in a 0.005 km2 unit gives 200 events/km2). The map
  therefore uses global Empirical Bayes smoothing (Marshall, 1991), which
  shrinks unreliable rates towards the valley mean in proportion to unit area.

Author: (thesis) -- generated as a working scaffold, verify before publishing.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

try:
    from scipy import stats as _st
except ImportError:  # scipy is optional; only the chi2 test needs it
    _st = None


# ---------------------------------------------------------------------------
# Shared constants (kept identical to eda_slope_unit_types.ipynb)
# ---------------------------------------------------------------------------
TYPES = [1, 2, 3]

SU_LABELS = {
    1: "Steep concave-convex slopes",
    2: "Intermediate concave-convex slopes",
    3: "Steep concave slopes",
}
SU_COLORS = {1: "#E69F00", 2: "#009E73", 3: "#D81B60"}   # Okabe-Ito
SU_SHORT = {1: "SU 1", 2: "SU 2", 3: "SU 3"}

# Verify these against Chapter 3 before publishing.
ZONE_LABELS = {1: "Oriente", 2: "Norte", 3: "Sur", 4: "SAP", 5: "Occidente"}

SEED = 42

DENSITY_CMAP = "YlOrRd"


# ===========================================================================
# 0. Preparation and diagnostics
# ===========================================================================
def prepare(gdf, type_col="su_type", count_col="n_mm", zone_col="zona_lluvia",
            verbose=True):
    """Validate the SU layer and add the columns the analysis needs.

    Returns a copy with `area_km2`, integer `su_type`, integer `zone_col`
    (nullable) and `count_col` with NaNs set to zero.

    Prints the diagnostics you must report in the thesis: totals, coverage,
    and any unit that would silently break the density computation.
    """
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("A projected CRS in metres is required (e.g. EPSG:3116).")

    g = gdf.copy()
    g = g[g[type_col].isin(TYPES)].copy()
    g[type_col] = g[type_col].astype(int)

    g["area_km2"] = g.geometry.area / 1e6

    n_missing_counts = int(g[count_col].isna().sum())
    g[count_col] = pd.to_numeric(g[count_col], errors="coerce").fillna(0.0)

    if zone_col in g.columns:
        g[zone_col] = (pd.to_numeric(g[zone_col], errors="coerce")
                         .round().astype("Int64"))

    n_zero_area = int((g["area_km2"] <= 0).sum())

    if verbose:
        print("=" * 68)
        print("SU x INVENTORY CROSS-ANALYSIS -- input diagnostics")
        print("=" * 68)
        print(f"Slope units                 : {len(g):,}")
        print(f"Total area                  : {g['area_km2'].sum():,.1f} km2")
        print(f"Total landslides assigned   : {int(g[count_col].sum()):,}")
        print(f"Units with >=1 landslide    : {int((g[count_col] > 0).sum()):,} "
              f"({100 * (g[count_col] > 0).mean():.1f} %)")
        print(f"Max landslides in one unit  : {int(g[count_col].max())}")
        if n_missing_counts:
            print(f"  !! {n_missing_counts} units had a missing count -> set to 0")
        if n_zero_area:
            print(f"  !! {n_zero_area} units have zero/negative area -> check geometry")
        if zone_col in g.columns:
            n_no_zone = int(g[zone_col].isna().sum())
            print(f"Units without rainfall zone : {n_no_zone:,}")
            print("Units per rainfall zone     :")
            for z, k in g[zone_col].value_counts().sort_index().items():
                print(f"    zone {z} ({ZONE_LABELS.get(int(z), '?')}): {k:,}")
        print("-" * 68)
        print("REPORT THESE IN THE THESIS and fill in by hand:")
        print("  * rule used to assign a landslide to a unit "
              "(centroid / intersection / crown point)")
        print("  * landslides discarded by the slope <= 10 deg filter "
              "(needs the pre-filter layer)")
        print("  * landslides falling outside the valley boundary")
        print("=" * 68)

    return g


# ===========================================================================
# 1. Traceability table + Frequency Ratio
# ===========================================================================
def traceability_table(gdf, type_col="su_type", count_col="n_mm"):
    """Per-type summary of the cross-analysis, with a TOTAL row.

    Columns
    -------
    n_units, pct_units, area_km2, pct_area, n_ls, pct_ls,
    density (events/km2), fr (frequency ratio), pct_affected
    """
    rows = []
    A = gdf["area_km2"].sum()
    N = gdf[count_col].sum()
    U = len(gdf)

    for t in TYPES:
        sub = gdf[gdf[type_col] == t]
        a_t = sub["area_km2"].sum()
        n_t = sub[count_col].sum()
        rows.append({
            "su_type": t,
            "label": SU_LABELS[t],
            "n_units": len(sub),
            "pct_units": 100 * len(sub) / U,
            "area_km2": a_t,
            "pct_area": 100 * a_t / A,
            "n_ls": n_t,
            "pct_ls": 100 * n_t / N if N else np.nan,
            "density": n_t / a_t if a_t else np.nan,
            "fr": (n_t / N) / (a_t / A) if (N and a_t) else np.nan,
            "pct_affected": 100 * (sub[count_col] > 0).mean(),
        })

    rows.append({
        "su_type": 0, "label": "All slope units",
        "n_units": U, "pct_units": 100.0,
        "area_km2": A, "pct_area": 100.0,
        "n_ls": N, "pct_ls": 100.0,
        "density": N / A if A else np.nan, "fr": 1.0,
        "pct_affected": 100 * (gdf[count_col] > 0).mean(),
    })

    tab = pd.DataFrame(rows).set_index("su_type")
    return tab


def bootstrap_ci(gdf, type_col="su_type", count_col="n_mm",
                 n_boot=2000, seed=SEED, chunk=100, alpha=0.05, verbose=True):
    """Bootstrap CIs for the per-type density and frequency ratio.

    Slope units are resampled WITH REPLACEMENT within each type (the type is
    fixed by design; what is uncertain is which units, and therefore how many
    landslides and how much area, fall in it). Both N_t and A_t are recomputed
    in every replicate, so the ratio structure of D and FR is preserved.

    Memory is bounded by processing `chunk` replicates at a time.

    Returns
    -------
    dict with keys 'density' and 'fr', each a DataFrame indexed by su_type with
    columns lo, hi (percentile interval) and se.
    """
    rng = np.random.default_rng(seed)

    n_sub, a_sub = {}, {}
    for t in TYPES:
        m = (gdf[type_col] == t).to_numpy()
        n_sub[t] = gdf.loc[m, count_col].to_numpy(dtype=float)
        a_sub[t] = gdf.loc[m, "area_km2"].to_numpy(dtype=float)

    Ns = np.empty((n_boot, len(TYPES)))
    As = np.empty((n_boot, len(TYPES)))

    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        for j, t in enumerate(TYPES):
            nt = n_sub[t].size
            idx = rng.integers(0, nt, size=(m, nt))
            Ns[start:start + m, j] = n_sub[t][idx].sum(axis=1)
            As[start:start + m, j] = a_sub[t][idx].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        D = Ns / As
        FR = (Ns / Ns.sum(axis=1, keepdims=True)) / (As / As.sum(axis=1, keepdims=True))

    qlo, qhi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for name, arr in (("density", D), ("fr", FR)):
        out[name] = pd.DataFrame({
            "lo": np.nanpercentile(arr, qlo, axis=0),
            "hi": np.nanpercentile(arr, qhi, axis=0),
            "se": np.nanstd(arr, axis=0, ddof=1),
        }, index=pd.Index(TYPES, name="su_type"))

    if verbose:
        print(f"Bootstrap: {n_boot} replicates, stratified by SU type, seed={seed}")
        print(f"  {int(100 * (1 - alpha))}% percentile intervals\n")
        print("  density (events/km2):")
        print(out["density"].round(3).to_string())
        print("\n  frequency ratio:")
        print(out["fr"].round(3).to_string())

    return out


def chi2_area_test(gdf, type_col="su_type", count_col="n_mm", verbose=True):
    """Test the null that landslides are distributed in proportion to type area.

    Returns dict with chi2, dof, p, cramers_v, and the per-type contributions
    (which show WHICH type drives the departure from the null).
    """
    if _st is None:
        raise ImportError("scipy is required for chi2_area_test")

    obs, exp, A, N = [], [], gdf["area_km2"].sum(), gdf[count_col].sum()
    for t in TYPES:
        sub = gdf[gdf[type_col] == t]
        obs.append(sub[count_col].sum())
        exp.append(N * sub["area_km2"].sum() / A)
    obs, exp = np.array(obs, float), np.array(exp, float)

    contrib = (obs - exp) ** 2 / exp
    chi2 = contrib.sum()
    dof = len(TYPES) - 1
    p = float(_st.chi2.sf(chi2, dof))   # survival function: accurate in the far tail
    cramers_v = np.sqrt(chi2 / (N * dof))

    res = {
        "chi2": chi2, "dof": dof, "p": p, "cramers_v": cramers_v,
        "observed": obs, "expected": exp,
        "contribution_pct": 100 * contrib / chi2,
    }

    if verbose:
        print(f"\nChi-square goodness of fit vs. area-proportional null")
        print(f"  chi2 = {chi2:,.1f}  (df = {dof})   "
              f"p {'< 1e-12' if p < 1e-12 else f'= {p:.3g}'}")
        print(f"  Cramer's V = {cramers_v:.3f}")
        for j, t in enumerate(TYPES):
            print(f"  {SU_SHORT[t]}: observed {obs[j]:>6.0f} | "
                  f"expected {exp[j]:>7.1f} | "
                  f"contributes {res['contribution_pct'][j]:5.1f}% of chi2")

    return res


def plot_frequency_ratio(tab, ci=None, figsize=(11.5, 4.8), savepath=None, dpi=300):
    """Two-panel Frequency Ratio figure.

    (a) share of valley area vs share of landslides, per type
    (b) frequency ratio with bootstrap CI and the FR = 1 reference line
    """
    t_rows = tab.loc[TYPES]
    x = np.arange(len(TYPES))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- (a) paired shares -------------------------------------------------
    w = 0.38
    b1 = ax1.bar(x - w / 2, t_rows["pct_area"], w, color="#BDBDBD",
                 edgecolor="#333", label="% of valley area")
    b2 = ax1.bar(x + w / 2, t_rows["pct_ls"], w,
                 color=[SU_COLORS[t] for t in TYPES],
                 edgecolor="#333", label="% of landslides")
    for bars in (b1, b2):
        for b in bars:
            ax1.annotate(f"{b.get_height():.1f}",
                         (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SU_SHORT[t] for t in TYPES])
    ax1.set_ylabel("Share of the total (%)")
    ax1.set_ylim(0, max(t_rows["pct_ls"].max(), t_rows["pct_area"].max()) * 1.18)
    ax1.grid(True, axis="y", ls="--", alpha=0.35)
    ax1.set_axisbelow(True)
    ax1.legend(handles=[Patch(facecolor="#BDBDBD", edgecolor="#333",
                              label="% of valley area"),
                        Patch(facecolor="#777777", edgecolor="#333",
                              label="% of landslides")],
               loc="upper right", fontsize=9)
    ax1.set_title("(a)", loc="left", fontsize=12)

    # --- (b) frequency ratio ----------------------------------------------
    fr = t_rows["fr"].to_numpy(float)
    if ci is not None:
        lo = fr - ci["fr"].loc[TYPES, "lo"].to_numpy(float)
        hi = ci["fr"].loc[TYPES, "hi"].to_numpy(float) - fr
        yerr = np.vstack([lo, hi])
    else:
        yerr = None

    bars = ax2.bar(x, fr, 0.6, color=[SU_COLORS[t] for t in TYPES],
                   edgecolor="#333", yerr=yerr, capsize=5,
                   error_kw=dict(ecolor="#333", lw=1.3))
    ax2.axhline(1.0, color="#333", lw=1.4, ls="--")
    ax2.annotate("FR = 1  (proportional to area)", (len(TYPES) - 0.45, 1.0),
                 xytext=(0, 5), textcoords="offset points",
                 ha="right", va="bottom", fontsize=9, color="#333")
    for j, b in enumerate(bars):
        top = b.get_height() + (yerr[1][j] if yerr is not None else 0)
        ax2.annotate(f"{fr[j]:.2f}", (b.get_x() + b.get_width() / 2, top),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SU_SHORT[t] for t in TYPES])
    ax2.set_ylabel("Frequency ratio (-)")
    ax2.set_ylim(0, max(1.35, (fr + (yerr[1] if yerr is not None else 0)).max() * 1.20))
    ax2.grid(True, axis="y", ls="--", alpha=0.35)
    ax2.set_axisbelow(True)
    ax2.set_title("(b)", loc="left", fontsize=12)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, facecolor="white", bbox_inches="tight")
    return fig


def table_to_latex(tab, ci=None, path=None,
                   caption=("Cross-analysis between the Slope Unit types and the "
                            "landslide inventory of the Aburra Valley."),
                   label="tab:su_inventory"):
    """Write the traceability table as a booktabs LaTeX table.

    Numbers are pre-formatted, so `escape=False` is safe.
    """
    bs = chr(92)          # a single backslash, avoids escaping noise below
    nl = bs + bs          # LaTeX end of row

    def thousands(v):
        return f"{int(round(v)):,}".replace(",", bs + ",")

    lines = []
    lines.append(bs + "begin{table}[H]")
    lines.append("    " + bs + "centering")
    lines.append("    " + bs + "footnotesize")
    lines.append("    " + bs + "begin{tabular}{lrrrrr}")
    lines.append("        " + bs + "hline")
    header = (bs + "textbf{Type} & " + bs + "textbf{Units (\\%)} & "
              + bs + "textbf{Area km$^{2}$ (\\%)} & "
              + bs + "textbf{Landslides (\\%)} & "
              + bs + "textbf{$D_t$ (ev.\\,km$^{-2}$)} & "
              + bs + "textbf{FR} " + nl)
    lines.append("        " + header)
    lines.append("        " + bs + "hline")

    for t in list(TYPES) + [0]:
        r = tab.loc[t]
        name = SU_SHORT[t] if t in TYPES else "All units"
        d_txt = f"{r['density']:.2f}"
        fr_txt = f"{r['fr']:.2f}"
        if ci is not None and t in TYPES:
            d_txt += f" [{ci['density'].loc[t, 'lo']:.2f}--{ci['density'].loc[t, 'hi']:.2f}]"
            fr_txt += f" [{ci['fr'].loc[t, 'lo']:.2f}--{ci['fr'].loc[t, 'hi']:.2f}]"
        row = (f"{name} & {thousands(r['n_units'])} ({r['pct_units']:.1f}) & "
               f"{r['area_km2']:.1f} ({r['pct_area']:.1f}) & "
               f"{thousands(r['n_ls'])} ({r['pct_ls']:.1f}) & "
               f"{d_txt} & {fr_txt} " + nl)
        if t == 0:
            lines.append("        " + bs + "hline")
        lines.append("        " + row)

    lines.append("        " + bs + "hline")
    lines.append("    " + bs + "end{tabular}")
    lines.append("    " + bs + "caption{" + caption + "}")
    lines.append("    " + bs + "label{" + label + "}")
    lines.append(bs + "end{table}")

    tex = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(tex + "\n")
    return tex


# ===========================================================================
# 2. SU type x rainfall zone heatmap
# ===========================================================================
def type_zone_matrix(gdf, type_col="su_type", count_col="n_mm",
                     zone_col="zona_lluvia"):
    """Cross-tabulate landslides, area, density and FR by (SU type, zone).

    Returns a dict of DataFrames indexed by SU type with rainfall zones as
    columns: 'n', 'area', 'density', 'fr', 'n_units', plus the marginals
    'by_type', 'by_zone' and the scalar 'global_density'.
    """
    g = gdf.dropna(subset=[zone_col]).copy()
    g[zone_col] = g[zone_col].astype(int)
    zones = [int(z) for z in sorted(g[zone_col].unique())]

    n_dropped = len(gdf) - len(g)
    if n_dropped:
        warnings.warn(f"{n_dropped} units have no rainfall zone and are excluded "
                      f"from the type x zone matrix. The marginals of this figure "
                      f"may therefore differ slightly from the totals of the "
                      f"traceability table.")

    n = g.pivot_table(index=type_col, columns=zone_col, values=count_col,
                      aggfunc="sum").reindex(index=TYPES, columns=zones)
    a = g.pivot_table(index=type_col, columns=zone_col, values="area_km2",
                      aggfunc="sum").reindex(index=TYPES, columns=zones)
    u = (pd.crosstab(g[type_col], g[zone_col])
           .reindex(index=TYPES, columns=zones)
           .astype(float))

    with np.errstate(divide="ignore", invalid="ignore"):
        d = n / a

    d_glob = g[count_col].sum() / g["area_km2"].sum()

    by_type = pd.DataFrame({
        "n": n.sum(axis=1), "area": a.sum(axis=1), "n_units": u.sum(axis=1)})
    by_type["density"] = by_type["n"] / by_type["area"]

    by_zone = pd.DataFrame({
        "n": n.sum(axis=0), "area": a.sum(axis=0), "n_units": u.sum(axis=0)})
    by_zone["density"] = by_zone["n"] / by_zone["area"]

    return {"n": n, "area": a, "n_units": u, "density": d, "fr": d / d_glob,
            "by_type": by_type, "by_zone": by_zone, "global_density": d_glob,
            "zones": zones}


def plot_type_zone_heatmap(mats, min_units=30, min_area_km2=1.0,
                           figsize=(11.5, 5.0), savepath=None, dpi=300,
                           cmap=DENSITY_CMAP, annotate_n=True):
    """Landslide density heatmap by SU type and rainfall zone, with margins.

    The right column and bottom row are the marginal densities (per type and
    per zone) drawn on the SAME colour scale, so the reader can see directly
    whether the variation is stronger across rows (morphology) or across
    columns (rainfall regime).

    Cells backed by fewer than `min_units` units or less than `min_area_km2`
    are masked and labelled 'n.d.', because their density is unreliable.
    """
    d, n, a, u = mats["density"], mats["n"], mats["area"], mats["n_units"]
    zones = mats["zones"]

    unreliable = (u.fillna(0) < min_units) | (a.fillna(0) < min_area_km2)
    d_plot = d.mask(unreliable)

    all_vals = np.concatenate([
        d_plot.to_numpy(float).ravel(),
        mats["by_type"]["density"].to_numpy(float).ravel(),
        mats["by_zone"]["density"].to_numpy(float).ravel(),
    ])
    vmax = float(np.nanmax(all_vals))
    norm = Normalize(vmin=0, vmax=vmax)

    nz, nt = len(zones), len(TYPES)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 4, width_ratios=[nz, 1.0, 0.22, 0.28], height_ratios=[nt, 1.0],
        wspace=0.10, hspace=0.10)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 0])
    ax_cnr = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[:, 2]).axis("off")
    cax = fig.add_subplot(gs[0, 3])

    def _draw(ax, mat, row_labels, col_labels, cnt=None, mask=None, fs=11):
        arr = np.asarray(mat, dtype=float)
        ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm,
                  aspect="auto", interpolation="nearest")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if not np.isfinite(v):
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="#EEEEEE", edgecolor="none",
                                           zorder=1))
                    ax.text(j, i, "n.d.", ha="center", va="center",
                            fontsize=9, color="#666", zorder=3)
                    continue
                col = "white" if v > 0.62 * norm.vmax else "#111111"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=fs, fontweight="bold", color=col)
                if cnt is not None and np.isfinite(np.asarray(cnt, float)[i, j]):
                    ax.text(j, i + 0.30, f"n={int(np.asarray(cnt, float)[i, j])}",
                            ha="center", va="center", fontsize=7.5, color=col)
        ax.set_xticks(range(arr.shape[1]))
        ax.set_yticks(range(arr.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#999")

    zlab = [f"{ZONE_LABELS.get(z, z)}" for z in zones]
    tlab = [SU_SHORT[t] for t in TYPES]

    _draw(ax_main, d_plot.to_numpy(float), tlab, zlab,
          cnt=n.to_numpy(float) if annotate_n else None)
    ax_main.set_xticklabels([])
    ax_main.set_title("Landslide density by SU type and rainfall zone "
                      "(events km$^{-2}$)", fontsize=12, pad=10)

    _draw(ax_right, mats["by_type"]["density"].to_numpy(float).reshape(-1, 1),
          tlab, ["All zones"],
          cnt=mats["by_type"]["n"].to_numpy(float).reshape(-1, 1) if annotate_n else None)
    ax_right.set_yticklabels([])
    ax_right.set_xticklabels([])
    ax_right.set_title("marginal", fontsize=9, color="#555", pad=6)

    _draw(ax_bot, mats["by_zone"]["density"].to_numpy(float).reshape(1, -1),
          ["All types"], zlab,
          cnt=mats["by_zone"]["n"].to_numpy(float).reshape(1, -1) if annotate_n else None)

    _draw(ax_cnr, np.array([[mats["global_density"]]]), ["All types"], ["All zones"])
    ax_cnr.set_yticklabels([])

    for ax in (ax_bot, ax_cnr):
        plt.setp(ax.get_xticklabels(), rotation=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Landslide density (events km$^{-2}$)", fontsize=10)

    if savepath:
        fig.savefig(savepath, dpi=dpi, facecolor="white", bbox_inches="tight")
    return fig


# ===========================================================================
# 3. Choropleth maps of landslide activity per Slope Unit
# ===========================================================================
def eb_rate_global(counts, area, verbose=True):
    """Global Empirical Bayes smoothing of per-unit rates (Marshall, 1991).

    Raw per-unit densities are dominated by small units: a single landslide in
    a 0.005 km2 unit yields 200 events/km2. The EB estimator shrinks each raw
    rate towards the valley-wide rate by a weight that grows with unit area,
    so large units keep their observed rate and small units are pulled towards
    the mean in proportion to how little evidence they carry.

        m   = sum(n) / sum(a)                       global rate
        s2  = sum(a * (r - m)^2)/sum(a) - m/mean(a) prior variance (>= 0)
        w_i = s2 / (s2 + m/a_i)                     shrinkage weight
        theta_i = w_i * r_i + (1 - w_i) * m

    Returns (theta, w, m, s2).
    """
    n = np.asarray(counts, dtype=float)
    a = np.asarray(area, dtype=float)
    if np.any(a <= 0):
        raise ValueError("All units must have a strictly positive area.")

    m = n.sum() / a.sum()
    r = n / a
    s2 = np.sum(a * (r - m) ** 2) / a.sum() - m / a.mean()
    s2 = max(float(s2), 0.0)

    if s2 == 0.0:
        warnings.warn("Estimated prior variance is zero: the EB map will be flat. "
                      "Fall back to the raw rates.")
        w = np.zeros_like(a)
    else:
        w = s2 / (s2 + m / a)

    theta = w * r + (1 - w) * m

    if verbose:
        print(f"Empirical Bayes smoothing (Marshall 1991)")
        print(f"  global rate m        = {m:.3f} events/km2")
        print(f"  prior variance s2    = {s2:.4f}")
        print(f"  shrinkage weight w   : median {np.median(w):.3f}, "
              f"range {w.min():.3f}-{w.max():.3f}")
        print(f"  raw rate  : max {r.max():,.1f} events/km2")
        print(f"  EB  rate  : max {theta.max():,.1f} events/km2")

    return theta, w, m, s2


def _add_scalebar(ax, length_km=5, loc=(0.06, 0.06), color="#111"):
    """Simple metric scale bar; assumes the axes are in metres."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = length_km * 1000.0
    xs = x0 + loc[0] * (x1 - x0)
    ys = y0 + loc[1] * (y1 - y0)
    ax.plot([xs, xs + L], [ys, ys], color=color, lw=3, solid_capstyle="butt",
            zorder=6)
    ax.text(xs + L / 2, ys + 0.012 * (y1 - y0), f"{length_km} km",
            ha="center", va="bottom", fontsize=9, color=color, zorder=6)


def _add_north(ax, loc=(0.93, 0.88), color="#111"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.06 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=color, zorder=6)


def plot_activity_maps(gdf, count_col="n_mm", zone_col="zona_lluvia",
                       zone_boundaries=None, simplify_m=0,
                       count_bins=(0, 1, 2, 3, 5), n_classes=6,
                       figsize=(13.5, 8.0), savepath=None, dpi=300,
                       cmap=DENSITY_CMAP, verbose=True):
    """Two-panel map of landslide activity per Slope Unit.

    (a) landslide count per unit, in classes (0 shown in light grey)
    (b) Empirical-Bayes-smoothed landslide density, quantile classes

    Parameters
    ----------
    zone_boundaries : GeoDataFrame, optional
        Rainfall-zone polygons to overlay. Build once with
        `zones = gdf.dissolve(by='zona_lluvia')` and reuse -- dissolving 61k
        polygons is slow.
    simplify_m : float
        Douglas-Peucker tolerance in metres applied only for drawing. 10-20 m
        cuts the rendering time substantially and is invisible at page scale.
        Leave at 0 for the final, publication-quality run.
    """
    g = gdf.copy()
    if simplify_m:
        g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)

    g["dens_raw"] = g[count_col] / g["area_km2"]
    theta, w, m, s2 = eb_rate_global(g[count_col].to_numpy(),
                                     g["area_km2"].to_numpy(), verbose=verbose)
    g["dens_eb"] = theta

    fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize)

    # --- (a) landslide counts ---------------------------------------------
    edges = list(count_bins) + [np.inf]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo == 0:
            labels.append("0")
        elif not np.isfinite(hi):
            labels.append(f"$\\geq$ {int(lo)}")
        elif hi - lo == 1:
            labels.append(f"{int(lo)}")
        else:
            labels.append(f"{int(lo)}-{int(hi) - 1}")

    # labels=False returns the 0-based bin index directly, which avoids a
    # Categorical -> float conversion that is version-dependent in pandas.
    g["cnt_class"] = pd.cut(g[count_col], bins=edges, right=False,
                            labels=False).astype(float)

    base = plt.get_cmap(cmap)
    colors = ["#E8E8E8"] + [base(v) for v in
                            np.linspace(0.25, 0.95, len(labels) - 1)]
    cmap_cnt = ListedColormap(colors)
    g.plot(column="cnt_class", ax=axa, cmap=cmap_cnt,
           norm=BoundaryNorm(np.arange(-0.5, len(labels)), len(labels)),
           linewidth=0, edgecolor="none", rasterized=True)
    axa.legend(handles=[Patch(facecolor=colors[i], edgecolor="#999",
                              label=labels[i]) for i in range(len(labels))],
               title="Landslides per\nslope unit", loc="upper left",
               fontsize=9, title_fontsize=10, frameon=True, framealpha=0.9)
    axa.set_title("(a) Landslide count per Slope Unit", loc="left", fontsize=12)

    # --- (b) EB-smoothed density ------------------------------------------
    pos = g.loc[g["dens_eb"] > 0, "dens_eb"]
    breaks = np.unique(np.nanquantile(pos, np.linspace(0, 1, n_classes + 1)))
    if breaks.size < 3:
        breaks = np.linspace(pos.min(), pos.max(), n_classes + 1)
    norm_eb = BoundaryNorm(breaks, len(breaks) - 1)

    g.plot(column="dens_eb", ax=axb, cmap=cmap, norm=norm_eb,
           linewidth=0, edgecolor="none", rasterized=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_eb)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axb, fraction=0.038, pad=0.02,
                      ticks=breaks, format="%.1f")
    cb.set_label("Smoothed landslide density (events km$^{-2}$)", fontsize=10)
    axb.set_title("(b) Empirical-Bayes smoothed density", loc="left", fontsize=12)

    for ax in (axa, axb):
        if zone_boundaries is not None:
            zone_boundaries.boundary.plot(ax=ax, color="#222", linewidth=0.7,
                                          zorder=5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        _add_scalebar(ax)
        _add_north(ax)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, facecolor="white", bbox_inches="tight")
    return fig, g


def concentration_summary(gdf, type_col="su_type", count_col="n_mm"):
    """How concentrated is the inventory? Useful sanity check for the map.

    Reports, per type and overall, the share of units that hold the inventory
    and the Gini coefficient of landslide counts across units. A very high
    concentration means the aggregate density is driven by a few units and
    should be stated explicitly in the text.
    """
    def _gini(x):
        x = np.sort(np.asarray(x, dtype=float))
        nn = x.size
        if nn == 0 or x.sum() == 0:
            return np.nan
        idx = np.arange(1, nn + 1)
        return float((2 * idx - nn - 1).dot(x) / (nn * x.sum()))

    rows = []
    for t in list(TYPES) + [0]:
        sub = gdf if t == 0 else gdf[gdf[type_col] == t]
        c = sub[count_col].to_numpy(float)
        order = np.sort(c)[::-1]
        top5 = order[:max(1, int(0.05 * c.size))].sum()
        rows.append({
            "su_type": t,
            "pct_units_affected": 100 * (c > 0).mean(),
            "pct_ls_in_top5pct_units": 100 * top5 / c.sum() if c.sum() else np.nan,
            "gini": _gini(c),
            "max_per_unit": int(c.max()) if c.size else 0,
        })
    return pd.DataFrame(rows).set_index("su_type")
