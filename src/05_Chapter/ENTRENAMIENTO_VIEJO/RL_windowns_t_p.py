import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

# -----------------------------
# 1. Cargar datos
# -----------------------------
path = "/home/oisanchezp/Thesis/data/processed/slope_units_si_no_20251115.csv"
df = pd.read_csv(path)

df = df[df["si_no"].isin([0, 1])].copy()
df = df.dropna(subset=["Mes", "ONI", "si_no"])

df["Mes"] = df["Mes"].astype(int)
df["si_no"] = df["si_no"].astype(int)

# -----------------------------
# 2. Función para construir T y P
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

    needed_cols = [col_T, col_tot, "Mes", "ONI", "si_no"]
    for c in needed_cols:
        if c not in df.columns:
            print(f"Falta la columna {c} en el DataFrame. Ajusta los nombres.")
            return None, None

    sub = df.dropna(subset=needed_cols).copy()
    if sub.empty:
        return None, None

    T_vals = sub[col_T].values
    P_vals = sub[col_tot].values - sub[col_T].values

    # Matriz de predictores: [T, P, Mes, ONI]
    X = np.column_stack([
        T_vals,
        P_vals,
        sub["Mes"].values.astype(float),
        sub["ONI"].values.astype(float)
    ])
    y = sub["si_no"].values.astype(int)

    return X, y

# -----------------------------
# 3. Función para evaluar un par (T,P) con regresión logística + CV
# -----------------------------
def evaluate_TP_with_cv_logit(df, t_days, p_days,
                              n_splits=5, n_repeats=5, random_state=42):
    """
    Devuelve lista de AUROC para la combinación (t_days, p_days)
    usando RepeatedStratifiedKFold con regresión logística.
    """
    X, y = build_TP_dataset(df, t_days, p_days)
    if X is None or y is None or len(np.unique(y)) < 2:
        return None, 0

    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    aucs = []

    # Pipeline: estandarización + regresión logística
    # class_weight='balanced' por si hay cierto desbalance si/no
    base_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced'
        )
    )

    for train_idx, test_idx in rkf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = base_model
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
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
        if t + p > max_days:
            continue

        print(f"Probando (logit) T={t} días, P={p} días...")
        aucs, n_obs = evaluate_TP_with_cv_logit(df, t, p)
        if aucs is None:
            print(f"   -> combinación T={t}, P={p} días sin datos suficientes.")
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

out_path = "/home/oisanchezp/Thesis/data/processed/resultados_TP_logit_5x5.csv"
res_df.to_csv(out_path, index=False)
print(f"Resultados guardados en: {out_path}")
