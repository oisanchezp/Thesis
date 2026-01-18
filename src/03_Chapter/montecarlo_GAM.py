# ============================================================
# MONTECARLO solo GAM + barrido de ventanas T (1..7) y P (8..90)
# - En cada corrida MC se reconstruye el dataset NO-eventos
# - Para cada combinación (T,P) se entrena/evalúa un GAM (1 split)
# - best_lam se elige SOLO con X_train, y_train
# - Al final: mean/median/std por (T,P) para cada métrica
# ============================================================
import os


import numpy as np
import pandas as pd

from pygam import LogisticGAM, s, f
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score,
    balanced_accuracy_score, average_precision_score
)

def construir_si_no(seed, 
                    su_si_path, pluv_meta, ruta_series,
                    oni_path,
                    umbral_1d=5, umbral_7d=10, umbral_90d=100):
    rng = np.random.default_rng(seed)


    # Umbrales de lluvia
    UMBRAL_1D  = umbral_1d
    UMBRAL_7D  = umbral_7d
    UMBRAL_90D = umbral_90d

    # ventanas en horas
    H1   = 24
    H7   = 7 * 24
    H90  = 90 * 24

    # -------------------------
    # 1) Cargar inventario de SÍ (Slope Units) y días excluidos ±30
    # -------------------------
    su_si = pd.read_csv(su_si_path, parse_dates=["fecha_hora_evento"])

    # Asegura que tenga si_no=1
    if "si_no" not in su_si.columns:
        su_si["si_no"] = 1
    else:
        su_si.loc[:, "si_no"] = 1  # por si acaso

    # Días a excluir (por día) ±30 alrededor de cada evento SÍ
    excluidas = pd.Index(
        np.unique(
            np.concatenate([
                pd.date_range(d.normalize() - pd.Timedelta(days=30),
                            d.normalize() + pd.Timedelta(days=30),
                            freq="D").date
                for d in su_si["fecha_hora_evento"]
            ])
        )
    )

    # -------------------------
    # 2) Precargar series y armar candidatos (fechas+pluviómetro)
    # -------------------------
    pluvios = pd.read_csv(pluv_meta)

    # Opcional: restringir a pluviómetros que ya usaste en su_si
    # (garantiza asignarles una Slope Unit existente)
    if "Codigo_pluvio_lluvia" in su_si.columns:
        codigos_usable = su_si["Codigo_pluvio_lluvia"].dropna().unique()
        pluvios = pluvios[pluvios["Codigo"].isin(codigos_usable)].copy()

    candidatos = []
    series_acum = {}   # guarda cumsum por código para reutilizar

    rng = np.random.default_rng(seed)

    for _, p in pluvios.iterrows():
        cod = p["Codigo"]
        fn  = f"H_Datos_Procesados_Est_{cod}.csv"
        path = os.path.join(ruta_series, fn)
        if not os.path.exists(path):
            continue

        # Leer serie horaria
        df = pd.read_csv(path, usecols=["Fecha", "P"], parse_dates=["Fecha"])
        serie_h = (df.groupby("Fecha")["P"].sum()
                    .sort_index()
                    .asfreq("H", fill_value=0.0))

        # Cumulada para diferencias rápidas
        cs = serie_h.cumsum().astype(np.float32)
        series_acum[cod] = cs

        idx = cs.index

        # Lluvias acumuladas 1d, 7d, 90d
        ll_1d  = cs - cs.shift(H1)
        ll_7d  = cs - cs.shift(H7)
        ll_90d = cs - cs.shift(H90)

        # Fuera de ventanas excluidas, comparando por DÍA
        fuera_ventanas = ~idx.normalize().isin(excluidas)

        # Candidatos que cumplen umbrales
        mask_ok = (
            fuera_ventanas &
            (ll_1d  >= UMBRAL_1D) &
            (ll_7d  >= UMBRAL_7D) &
            (ll_90d >= UMBRAL_90D)
        )

        fechas_ok = idx[mask_ok]
        if len(fechas_ok):
            # Guardar (timestamp, código) como candidatos
            candidatos.extend([(f, cod) for f in fechas_ok])

    # Quitar duplicados si los hubiera (mismo timestamp+código)
    candidatos = list({(f, c) for (f, c) in candidatos})

    # -------------------------
    # 3) Muestreo: 2× eventos SÍ
    # -------------------------
    n_si = len(su_si)
    n_no_target = 2 * n_si

    if len(candidatos) < n_no_target:
        raise ValueError(
            f"No hay suficientes candidatos que cumplan umbrales y ventanas: "
            f"necesitas {n_no_target}, hay {len(candidatos)}."
        )

    muestras = rng.choice(candidatos, size=n_no_target, replace=False)

    # -------------------------
    # 4) Preparar estructura de columnas: estáticas vs dinámicas
    # -------------------------
    lluvia_cols = [f"{d}d" for d in range(1, 98)]
    dynamic_cols = set(["fecha_hora_evento", "Mes", "DoY", "ONI", "ENSO", "si_no"] + lluvia_cols)

    # columnas estáticas (se copian de una SU real asociada al pluviómetro)
    static_cols = [c for c in su_si.columns if c not in dynamic_cols]

    # -------------------------
    # 5) Construir NO-eventos con acumulados
    # -------------------------
    filas = []

    for fecha, cod in muestras:
        cs = series_acum[cod]

        # Asegurarse de que la fecha esté en el índice (o la anterior)
        if fecha not in cs.index:
            try:
                pos = cs.index.get_loc(fecha, method="pad")
            except KeyError:
                # si no encuentra nada, saltamos este candidato
                continue
        else:
            pos = cs.index.get_loc(fecha)

        # Necesitamos al menos 97 días (97*24 h) hacia atrás
        if pos < 97 * H1:
            continue

        # Acumulados 1–97 días hacia atrás
        vals = cs.iloc[pos] - cs.iloc[pos - np.arange(1, 98) * H1].values

        # Buscar una Slope Unit asociada a este pluviómetro
        if "Codigo_pluvio_lluvia" in su_si.columns:
            su_candidatas = su_si[su_si["Codigo_pluvio_lluvia"] == cod]
        else:
            su_candidatas = su_si

        if su_candidatas.empty:
            # No tenemos SU asociadas a este pluvio, saltamos
            continue

        # Elegimos una SU al azar entre las que usan ese pluviómetro
        base = su_candidatas.sample(1, random_state=rng.integers(0, 10**9)).iloc[0]

        fila = {}

        # Copiamos todas las columnas estáticas desde esa SU
        for col in static_cols:
            fila[col] = base[col]

        # Dinámicas redefinidas
        fila["fecha_hora_evento"] = pd.Timestamp(fecha)
        fila["si_no"] = 0
        fila["Mes"] = fecha.month
        fila["DoY"] = fecha.dayofyear

        # Código de pluviómetro usado para lluvia (si existe en columnas estáticas, lo sobreescribimos)
        if "Codigo_pluvio_lluvia" in su_si.columns:
            fila["Codigo_pluvio_lluvia"] = cod
        else:
            fila["Codigo"] = cod

        # Lluvia 1–97 días
        for d in range(1, 98):
            fila[f"{d}d"] = float(vals[d-1])

        filas.append(fila)

    df_no = pd.DataFrame(filas)

    # Reaplica filtro de umbrales por si alguna fecha quedó al borde de la serie
    df_no = df_no[
        (df_no["1d"]  >= UMBRAL_1D) &
        (df_no["7d"]  >= UMBRAL_7D) &
        (df_no["90d"] >= UMBRAL_90D)   # si usas 90d explícito; si no, ajusta a 91d
    ].copy()

    # Si tras la verificación hay más de lo necesario, recorta aleatoriamente al objetivo
    if len(df_no) > n_no_target:
        df_no = df_no.sample(n=n_no_target, random_state=seed)

    # -------------------------
    # 6) Añadir ONI y ENSO a los NO-eventos
    # -------------------------
    oni_df = pd.read_csv(oni_path, parse_dates=["date"])

    oni_df["date"] = pd.to_datetime(oni_df["date"]).dt.normalize()
    oni_max = oni_df["date"].max()
    oni_min = oni_df["date"].min()
    su_si = su_si[su_si["fecha_hora_evento"].dt.normalize().between(oni_min, oni_max)].copy()

    df_no["date"] = df_no["fecha_hora_evento"].dt.normalize()
    df_no = df_no.merge(
        oni_df[["date", "ONI", "ENSO"]],
        on="date",
        how="left"
    ).drop(columns=["date"])

    df_no = df_no[df_no["fecha_hora_evento"].dt.normalize().between(oni_min, oni_max)].copy()

    # -------------------------
    # 7) Unir SI y NO y guardar
    # -------------------------
    df_si = su_si.copy()

    # Nos aseguramos de que df_no tenga todas las columnas de df_si; las que falten se rellenan con NaN
    for col in df_si.columns:
        if col not in df_no.columns:
            df_no[col] = np.nan

    # Alineamos orden de columnas
    df_no = df_no[df_si.columns]

    df_combined = pd.concat([df_si, df_no], axis=0, ignore_index=True)

    print("Nulos ENSO:", df_combined["ENSO"].isna().sum())
    print("Valores únicos ENSO (muestra):", df_combined["ENSO"].astype(str).str.strip().value_counts().head(20))

    enso_map = {
        "La Niña": 0,
        "Neutro": 1,
        "El Niño": 2,
    }
    df_combined["ENSO_code"] = df_combined["ENSO"].map(enso_map).astype(int)
    print("No mapeados:", df_combined.loc[df_combined["ENSO_code"].isna(), "ENSO"].astype(str).str.strip().value_counts().head(20))
    # Ángulo en radianes
    df_combined["doy_rad"] = 2 * np.pi * (df_combined["DoY"] - 1) / 365.0
    df_combined["sin_doy"] = np.sin(df_combined["doy_rad"])
    df_combined["cos_doy"] = np.cos(df_combined["doy_rad"])
    #df_combined.to_csv(out_path, index=False)

    print(f"Eventos SÍ: {len(df_si)}  |  NO-eventos: {len(df_no)}  |  Total: {len(df_combined)}")
    #print(f"Guardado en: {out_path}")

    return df_combined


# ---------------------------------------------------
# 1) Términos del GAM: X = [T, P, ENSO_code, sin_doy, cos_doy]
# ---------------------------------------------------
terms = (
    s(0, n_splines=5) +   # T
    s(1, n_splines=5) +   # P
    s(3, n_splines=5) +   # sin_doy
    s(4, n_splines=5) +   # cos_doy
    f(2)                  # ENSO_code (categórica)
)

lam_grid = np.logspace(0, 3, 5)  # 1, 3.16, 10, 31.6, 1000 aprox.


def get_best_lambda_train_only(X_train, y_train, random_state=42, val_size=0.30):
    """Elige best_lam usando SOLO train (split interno train/val dentro del train)."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )

    best_lam = lam_grid[0]
    best_auc = -np.inf

    for lam in lam_grid:
        gam = LogisticGAM(terms, lam=lam).fit(X_tr, y_tr)
        y_prob = gam.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_prob)
        if auc > best_auc:
            best_auc = auc
            best_lam = lam

    return best_lam


def eval_metrics(y_true, y_prob, thr=0.5):
    """Métricas basadas en probas + umbral thr."""
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0          # POD
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0       # PPV
    pofd = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0            # POFD (FPR)
    hk = (recall - pofd) if ((tp + fn) > 0 and (fp + tn) > 0) else 0.0

    return {
        "auc":  float(roc_auc_score(y_true, y_prob)),
        "ap":   float(average_precision_score(y_true, y_prob)),   # PR-AUC
        "f1":   float(f1_score(y_true, y_pred, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),

        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),

        "recall": float(recall),          # POD
        "precision": float(precision),    # PPV
        "pofd": float(pofd),              # POFD
        "hk": float(hk),                  # HK = POD - POFD
    }


def prepare_sub_base(df_mc, max_days=97):
    """
    Prepara una tabla base con TODAS las columnas que podrías necesitar
    para cualquier (T,P) con T in [1..7] y P in [8..90] (total <= 97).
    """
    rain_cols = [f"{d}d" for d in range(1, max_days + 1)]

    needed = ["si_no", "ENSO_code", "sin_doy", "cos_doy"] + rain_cols
    missing = [c for c in needed if c not in df_mc.columns]
    if missing:
        raise ValueError(f"Faltan columnas en df_mc: {missing}")

    sub = df_mc[needed].copy()

    # Asegurar numéricos y quitar NA
    sub["ENSO_code"] = pd.to_numeric(sub["ENSO_code"], errors="coerce")
    for c in ["sin_doy", "cos_doy"] + rain_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub = sub.dropna().copy()

    if sub.empty or sub["si_no"].nunique() < 2:
        raise ValueError("No hay datos suficientes (o falta una clase) tras limpiar NA.")

    sub["si_no"] = sub["si_no"].astype(int)
    sub["ENSO_code"] = sub["ENSO_code"].astype(int)

    return sub


def build_Xy_from_sub(sub, t_days, p_days):
    """
    Construye X,y para una combinación (T=t_days, P=p_days).
    P = R_{t+p} - R_t.
    """
    total = t_days + p_days
    col_t = f"{t_days}d"
    col_tot = f"{total}d"

    if col_t not in sub.columns or col_tot not in sub.columns:
        raise ValueError(f"Faltan columnas {col_t} o {col_tot} para T={t_days}, P={p_days}")

    T = sub[col_t].values
    P = (sub[col_tot].values - sub[col_t].values)

    X = np.column_stack([
        T,
        P,
        sub["ENSO_code"].values.astype(int),
        sub["sin_doy"].values.astype(float),
        sub["cos_doy"].values.astype(float),
    ])
    y = sub["si_no"].values.astype(int)

    return X, y


def evaluate_gam_once_for_TP(sub_base, t_days, p_days, test_size=0.30, random_state=42, thr=0.5):
    """
    Evalúa SOLO GAM (split único).
    best_lam se estima SOLO con X_train,y_train.
    """
    X, y = build_Xy_from_sub(sub_base, t_days, p_days)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    best_lam = get_best_lambda_train_only(X_train, y_train, random_state=random_state)

    gam = LogisticGAM(terms, lam=best_lam).fit(X_train, y_train)
    y_prob = gam.predict_proba(X_test)

    m = eval_metrics(y_test, y_prob, thr=thr)
    m["best_lam"] = float(best_lam)
    m["n_obs"] = int(len(y))
    m["n_train"] = int(len(y_train))
    m["n_test"] = int(len(y_test))

    return m


# ============================================================
# 2) PARÁMETROS DEL BARRIDO
# ============================================================
max_days = 97
short_terms = range(1, 8)    # T = 1..7
long_terms  = range(8, 91)   # P = 8..90 (longitud de ventana preparatoria)

# ============================================================
# 3) MONTECARLO
#    OJO: este bloque asume que YA tienes definida construir_si_no(...)
# ============================================================

su_si_path  = "/home/oisanchezp/Thesis/data/processed/slope_units_eventos.csv"
pluv_meta   = "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv"
ruta_series = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"
oni_path    = "/home/oisanchezp/Thesis/data/metadata/oni_diario_2010_2025.csv"

n_mc = 10
thr = 0.5

registros = []

for mc in range(n_mc):
    seed = 100 + mc
    print(f"\n=== Monte Carlo {mc+1}/{n_mc} (seed={seed}) ===")

    # 1) dataset nuevo (SÍ + NO)
    df_mc = construir_si_no(
        seed=seed,
        su_si_path=su_si_path,
        pluv_meta=pluv_meta,
        ruta_series=ruta_series,
        oni_path=oni_path,
        umbral_1d=5,
        umbral_7d=10,
        umbral_90d=100
    )

    # 2) tabla base limpia para reutilizar en todas las combinaciones T,P
    try:
        sub_base = prepare_sub_base(df_mc, max_days=max_days)
    except ValueError as e:
        print(f"   -> MC {mc} se salta: {e}")
        continue

    # 3) barrido (T,P)
    for t in short_terms:
        for p in long_terms:
            print(f"\n=== Evaluando T={t}, P={p}...de {mc+1}/{n_mc} ===")
            if (t + p) > max_days:
                continue

            try:
                met = evaluate_gam_once_for_TP(
                    sub_base, t_days=t, p_days=p,
                    test_size=0.30,
                    random_state=42,
                    thr=thr
                )
            except Exception as e:
                # si alguna combinación falla, la saltas sin matar todo
                print(f"   -> falla T={t}, P={p}: {e}")
                continue

            met["mc_id"] = mc
            met["T_days"] = int(t)
            met["P_days"] = int(p)
            met["threshold"] = float(thr)
            registros.append(met)

# ============================================================
# 4) RESULTADOS CRUDOS + RESUMEN (mean/median/std) por (T,P)
# ============================================================
results_df = pd.DataFrame(registros)

out_raw = "/home/oisanchezp/Thesis/data/processed/GAM_MC_raw_T1-7_P8-90.csv"
results_df.to_csv(out_raw, index=False)
print(f"\nGuardado RAW en: {out_raw}")

metric_cols = ["auc", "ap", "f1", "bacc", "recall", "precision", "pofd", "hk"]

summary_df = (
    results_df
    .groupby(["T_days", "P_days"])[metric_cols]
    .agg(["mean", "median", "std"])
    .reset_index()
)

out_sum = "/home/oisanchezp/Thesis/data/processed/GAM_MC_summary_T1-7_P8-90.csv"
summary_df.to_csv(out_sum, index=False)
print(f"Guardado SUMMARY en: {out_sum}")

# Si quieres ver top-10 por alguna métrica (ej: hk_mean)
# OJO: con columnas multi-index por el agg, quedan como ('hk','mean'), etc.
summary_df_sorted = summary_df.sort_values(("hk", "mean"), ascending=False).head(10)
print("\nTop 10 combinaciones por HK (mean):")
print(summary_df_sorted[["T_days", "P_days", ("hk", "mean"), ("auc", "mean"), ("ap", "mean")]])
