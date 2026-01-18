import pandas as pd
import numpy as np

from pygam import LogisticGAM, f, s, te
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

# -----------------------------
# 1. Cargar datos
# -----------------------------
path = "/home/oisanchezp/Thesis/data/processed/slope_units_si_no_20251115.csv"
df = pd.read_csv(path)

df = df[df["si_no"].isin([0, 1])].copy()
df = df.dropna(subset=["DoY", "ENSO", "si_no"])

enso_map = {
    "La Niña": 0,
    "Neutro": 1,
    "El Niño": 2,
}
df["ENSO_code"] = df["ENSO"].map(enso_map).astype(int)


df["DoY"] = df["DoY"].astype(int)
df["si_no"] = df["si_no"].astype(int)

# Ángulo en radianes
df["doy_rad"] = 2 * np.pi * (df["DoY"] - 1) / 365.0
df["sin_doy"] = np.sin(df["doy_rad"])
df["cos_doy"] = np.cos(df["doy_rad"])

# -----------------------------
# 2. Configuración global del GAM
# -----------------------------
# 0: T (corto plazo), 1: P (largo plazo), 2: Mes, 3: ONI
# terms = (
#     s(0, n_splines=5) +
#     s(1, n_splines=5) +
#     s(2, n_splines=5) +
#     s(3, n_splines=5)
# )

terms = (
    s(0, n_splines=10) +   # T
    s(1, n_splines=10) +   # P
    s(2, n_splines=5) +    # sin(DOY)
    s(3, n_splines=5) +    # cos(DOY)
    f(4)              #+  # ENSO como factor (La Niña, Neutro, El Niño)
    # te(0, 2) +          # interacción T y sin(DOY)
    # te(0, 3) +          # interacción T y cos(DOY)
    # te(1, 2) +          # interacción P y sin(DOY)
    # te(1, 3)            # interacción P y cos(DOY) 
)

lam_grid = np.logspace(0, 3, 5)  # 10 .. 1000

# -----------------------------
# 3. Funciones auxiliares
# -----------------------------
def build_TP_dataset(df, t_days, p_days, max_days=98):
    """
    t_days: longitud de la ventana de corto plazo T (1-7).
    p_days: longitud de la ventana de largo plazo P (8-90).
    max_days: máximo acumulado disponible en el DataFrame (90 por defecto).

    Asumimos columnas "1d", "2d", ..., "90d".
    T = R_t
    P = R_{t+p} - R_t
    """
    total_days = t_days + p_days
    if total_days > max_days:
        return None, None

    col_T = f"{t_days}d"
    col_tot = f"{total_days}d"

    needed_cols = [col_T, col_tot, "sin_doy", "cos_doy", "ENSO_code", "si_no"]
    for c in needed_cols:
        if c not in df.columns:
            print(f"Falta la columna {c} en el DataFrame. Ajusta los nombres.")
            return None, None

    sub = df.dropna(subset=needed_cols).copy()
    if sub.empty:
        return None, None

    T_vals = sub[col_T].values
    P_vals = sub[col_tot].values - sub[col_T].values

    X = np.column_stack([
        T_vals,
        P_vals,
        sub["sin_doy"].values.astype(float),
        sub["cos_doy"].values.astype(float),
        sub["ENSO_code"].values.astype(float)
    ])
    y = sub["si_no"].values.astype(int)

    return X, y


def get_best_lambda(X, y):
    """
    Selecciona el mejor lambda usando un solo split train/val rápido.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    scores = []
    for lam in lam_grid:
        gam = LogisticGAM(terms, lam=lam)
        gam.fit(X_train, y_train)
        y_prob = gam.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_prob)
        scores.append(auc)

    best_lam = lam_grid[np.argmax(scores)]
    print(f"   -> Mejor lambda: {best_lam} con AUC={max(scores):.4f}")
    return best_lam


def evaluate_TP_with_cv(df, t_days, p_days,
                        n_splits=10, n_repeats=5, random_state=42):
    """
    Devuelve lista de AUROC para la combinación (t_days, p_days)
    usando RepeatedStratifiedKFold.
    """
    X, y = build_TP_dataset(df, t_days, p_days)
    if X is None or y is None or len(np.unique(y)) < 2:
        return None, 0

    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    best_lam = get_best_lambda(X, y)
    aucs = []

    for train_idx, test_idx in rkf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        gam = LogisticGAM(terms, lam=best_lam).fit(X_train, y_train)
        y_prob = gam.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob)
        aucs.append(auc)

    return np.array(aucs), len(y)

# -----------------------------
# 4. Loop sobre todas las combinaciones T, P
# -----------------------------
results = []

max_days = 98
short_terms = range(1, 8)   # T = 1..7 días
long_terms  = range(8, 91)  # P = 8..90 días

for t in short_terms:
    for p in long_terms:
        print(f"Probando T={t} días, P={p} días...")
        aucs, n_obs = evaluate_TP_with_cv(df, t, p)
        if aucs is None:
            print(f"   -> combinación T={t}, P={p} sin datos suficientes.")
            continue

        results.append({
            "T_days": t,
            "P_days": p,
            "n_obs": n_obs,
            "median_auc": np.median(aucs),
            "mean_auc": np.mean(aucs),
            "std_auc": np.std(aucs)
        })

# -----------------------------
# 5. Resultados finales
# -----------------------------
res_df = pd.DataFrame(results).sort_values("median_auc", ascending=False)

print(res_df.head(10))

out_path = "/home/oisanchezp/Thesis/data/processed/resultados_TP_gam_10x5_2.csv"
res_df.to_csv(out_path, index=False)
print(f"Resultados guardados en: {out_path}")
