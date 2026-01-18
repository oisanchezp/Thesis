import os
import numpy as np
import pandas as pd

from pygam import LogisticGAM, s, f
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
# 1. Dataset T (1 día) y P (33 días)
# ---------------------------------------------------
def build_TP_dataset(df, t_days=1, p_days=33, max_days=97):
    """
    Construye X, y para una combinación fija de T y P.
    T = R_t
    P = R_{t+p} - R_t
    Donde R_k = lluvia acumulada en k días (columna 'kd').
    """
    total_days = t_days + p_days
    if total_days > max_days:
        raise ValueError(f"T+P = {total_days} > max_days={max_days}")

    col_T = f"{t_days}d"
    col_tot = f"{total_days}d"

    needed_cols = [col_T, col_tot, "Mes", "ONI", "si_no"]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Falta la columna {c} en el DataFrame.")

    sub = df.dropna(subset=needed_cols).copy()
    if sub.empty or sub["si_no"].nunique() < 2:
        raise ValueError("No hay datos suficientes o si_no no tiene ambas clases.")

    # T y P
    T_vals = sub[col_T].values
    P_vals = sub[col_tot].values - sub[col_T].values

    # Matriz de predictores: [T, P, Mes, ONI]
    X = np.column_stack([
        T_vals,
        P_vals,
        sub["ENSO_code"].values.astype(int),
        sub["sin_doy"].values.astype(float),
        sub["cos_doy"].values.astype(float)
    ])
    y = sub["si_no"].values.astype(int)

    return X, y

# ---------------------------------------------------
# 2. Configuración del GAM (semi-paramétrico)
# ---------------------------------------------------
# 0: T (corto plazo), 1: P (largo plazo), 2: Mes, 3: ONI
terms = (
    s(0, n_splines=5) +
    s(1, n_splines=5) +
    s(3, n_splines=5) +
    s(4, n_splines=5) +
    f(2)
)

lam_grid = np.logspace(0, 3, 5)  # 1, 3.16, 10, 31.6, 1000 aprox.


def get_best_lambda(X, y):
    """
    Selecciona el mejor lambda para el GAM con un solo split train/val.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    best_lam = lam_grid[0]
    best_auc = -np.inf

    for lam in lam_grid:
        gam = LogisticGAM(terms, lam=lam)
        gam.fit(X_train, y_train)
        y_prob = gam.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_prob)
        if auc > best_auc:
            best_auc = auc
            best_lam = lam

    return best_lam

# ---------------------------------------------------
# 3. Evaluar los 3 modelos para un solo dataset (una corrida MC)
#    SIN validación cruzada: un solo split train/test
#    best_lam se estima SOLO con X_train, y_train
# ---------------------------------------------------

def evaluate_models_once(df_mc,
                         t_days=1,
                         p_days=33,
                         test_size=0.30,
                         random_state=42,
                         thr=0.5):
    """
    Entrena y evalúa 3 modelos con un único split estratificado train/test:
    - GAM
    - Regresión Logística
    - Random Forest

    Devuelve 3 dicts (uno por modelo) con AUC, AP(PR-AUC), F1, BAcc y TP/TN/FP/FN.
    """
    X, y = build_TP_dataset(df_mc, t_days=t_days, p_days=p_days)

    # Split único (esto reemplaza la validación cruzada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    def eval_once(y_true, y_prob, thr=0.5):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "auc":  float(roc_auc_score(y_true, y_prob)),
            "ap":   float(average_precision_score(y_true, y_prob)),  # PR-AUC
            "f1":   float(f1_score(y_true, y_pred, zero_division=0)),
            "bacc": float(balanced_accuracy_score(y_true, y_pred)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "pofa": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "HK": float((tp / (tp + fn)) - (fp / (fp + tn))) if (tp + fn) > 0 and (fp + tn) > 0 else 0.0
        }

    def resumen(model_name, metrics):
        return {
            "modelo": model_name,
            "n_obs": int(len(y)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_folds": 1,
            "threshold": float(thr),
            "auc": metrics["auc"],
            "ap": metrics["ap"],
            "f1": metrics["f1"],
            "bacc": metrics["bacc"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "pofa": metrics["pofa"],
            "HK": metrics["HK"],
        }

    # ---------- GAM: best_lam SOLO con TRAIN ----------
    best_lam = get_best_lambda(X_train, y_train)
    print(f"Entrenando GAM (lam={best_lam})...")
    gam = LogisticGAM(terms, lam=best_lam).fit(X_train, y_train)
    y_prob_gam = gam.predict_proba(X_test)
    met_gam = eval_once(y_test, y_prob_gam, thr=thr)

    # ---------- Logit ----------
    print("Entrenando Regresión Logística...")
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced'
        )
    )
    logit.fit(X_train, y_train)
    y_prob_logit = logit.predict_proba(X_test)[:, 1]
    met_logit = eval_once(y_test, y_prob_logit, thr=thr)

    # ---------- Random Forest ----------
    print("Entrenando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight='balanced',
        random_state=0
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    met_rf = eval_once(y_test, y_prob_rf, thr=thr)

    return [
        resumen("GAM", met_gam),
        resumen("Logit", met_logit),
        resumen("RandomForest", met_rf),
    ]

# ---------------------------------------------------
# 4. Bucle Monte Carlo: 50 corridas
#    cambiando los NO-eventos con construir_si_no
# ---------------------------------------------------
su_si_path  = "/home/oisanchezp/Thesis/data/processed/slope_units_eventos.csv"
pluv_meta   = "/home/oisanchezp/Thesis/data/metadata/pluviometros_metadatos_20250506.csv"
ruta_series = "/mnt/investigacion/geotecnia/Pluvios_horarios/Pluvios_horarios_completos/"
oni_path    = "/home/oisanchezp/Thesis/data/metadata/oni_diario_2010_2025.csv"

n_mc = 100
registros = []

for mc in range(n_mc):
    seed = 100 + mc
    print(f"\n=== Monte Carlo {mc+1}/{n_mc} (seed={seed}) ===")

    # construyes un nuevo dataset con Sí + NO eventos
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

    try:
        resultados_mc = evaluate_models_once(
            df_mc,
            t_days=1,
            p_days=33,
            random_state=42,
            thr=0.5
        )
    except ValueError as e:
        print(f"   -> Corrida {mc} sin datos suficientes: {e}")
        continue

    for r in resultados_mc:
        r["mc_id"] = mc
        registros.append(r)

# ---------------------------------------------------
# 5. DataFrame final con métricas de los 3 modelos
# ---------------------------------------------------
results_df = pd.DataFrame(registros)

out_path = "/home/oisanchezp/Thesis/data/processed/resultados_1d_33d_GAM_Logit_RF_MC50.csv"
results_df.to_csv(out_path, index=False)

print(f"\nResultados guardados en: {out_path}")
print("\nResumen por modelo (AUROC medio sobre las 50 corridas):")
print(results_df.groupby("modelo")["auc"].describe())
