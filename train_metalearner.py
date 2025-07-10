# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Global Meta-Model Stacking with LightGBM, MLP and Linear Regression
# Description : 
#   • Loads station-level model predictions
#   • Constructs global dataset
#   • Adds basic temporal features
#   • Compares different meta-models using cross-validation with hyperparameter tuning
# Author      : Pablo Catret
# =============================================================================

# =============================================================================
# Import libraries
# =============================================================================
import pandas as pd
import numpy as np
import glob, re, warnings, joblib
from pathlib import Path
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# Step 1: Locate validation prediction files
# =============================================================================
CARPETA   = Path("SeleccionHiperparametros/Predicciones")
csv_files = glob.glob(str(CARPETA / "*_VAL.csv"))
pattern   = re.compile(r"(.+?)_([^_]+)_VAL\.csv")

# =============================================================================
# Step 2: Load and merge predictions by station
# =============================================================================

def cargar_datos_estacion(estacion):
    """Carga y fusiona los CSV de una estación en un único DataFrame"""
    archivos = [f for f in csv_files if Path(f).name.startswith(estacion)]
    dfs = []
    for f in archivos:
        modelo = pattern.search(Path(f).name).group(2)     # ETS, LSTM, …
        df     = pd.read_csv(f, parse_dates=["time"])
        dfs.append(df.rename(columns={"pred": f"pred_{modelo}"})
                     [["time", f"pred_{modelo}", "real"]])
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on=["time", "real"], how="inner")
    return base.sort_values("time").reset_index(drop=True)

estaciones = sorted({pattern.search(Path(f).name).group(1) for f in csv_files})

# =============================================================================
# Step 3: Build global dataset
# =============================================================================
dflist = []
for est in estaciones:
    dflist.append(cargar_datos_estacion(est))

df = pd.concat(dflist, axis=0, ignore_index=True)

# =============================================================================
# Step 4: Add temporal features (station not included)
# =============================================================================
df["month"]     = df.time.dt.month
df["quarter"]   = df.time.dt.quarter
df["dayofyear"] = df.time.dt.dayofyear
df["season"]    = ((df.month % 12) // 3).astype(int)

# =============================================================================
# Step 5: Define input feature sets
# =============================================================================
pred_cols   = [c for c in df.columns if c.startswith("pred_")]

# Solo incluimos una feature temporal básica para este ejemplo.
# Ajusta aquí si quieres añadir más variables (por ejemplo quarter, season…)
FEATURE_SETS = {
    "preds_only"  : pred_cols,
    "preds_temp"  : pred_cols + ["month"],
}

TARGET = "real"

# Configuración del K‑Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def evaluar(modelo, X, y):
    """Calcula RMSE y MAPE mediante validación cruzada K‑Fold"""
    rmses, mapes = [], []
    for tr_idx, te_idx in kf.split(X):
        modelo.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        y_hat = modelo.predict(X.iloc[te_idx])
        rmses.append(root_mean_squared_error(y.iloc[te_idx], y_hat))
        mapes.append(mean_absolute_percentage_error(y.iloc[te_idx], y_hat))
    return np.mean(rmses), np.mean(mapes)

# =============================================================================
# Step 6A: Randomized hyperparameter space for LightGBM
# =============================================================================
param_dist_lgb = {
    "num_leaves"        : np.arange(16, 128, 8),
    "max_depth"         : [-1, 3, 4, 5, 6, 7, 8],
    "learning_rate"     : np.logspace(-2, -0.7, 15),
    "subsample"         : np.linspace(0.6, 1.0, 5),
    "colsample_bytree"  : np.linspace(0.6, 1.0, 5),
    "min_child_samples" : np.arange(5, 60, 5),
    "lambda_l1"         : np.linspace(0.0, 1.0, 11),
    "lambda_l2"         : np.linspace(0.0, 1.0, 11),
}

# =============================================================================
# Step 6B: Randomized hyperparameter space for MLPRegressor
# =============================================================================
param_dist_mlp = {
    "hidden_layer_sizes" : [(64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
    "activation"         : ["relu", "tanh"],
    "solver"             : ["adam", "sgd"],
    "alpha"              : np.logspace(-6, -2, 5),
    "learning_rate_init" : np.logspace(-4, -2, 10),
    "batch_size"         : [32, 64, 128],
}

# =============================================================================
# Step 7: Train and evaluate meta-models
# =============================================================================
resultados  = []
mejor_rmse  = np.inf
mejor_modelo = None
mejor_set    = None

for set_name, cols in FEATURE_SETS.items():
    X = df[cols]
    y = df[TARGET]

    # -------------------- Linear Regression --------------------
    lr = LinearRegression()
    rmse_lr, mape_lr = evaluar(lr, X, y)
    resultados.append((set_name, "LinearReg", rmse_lr, mape_lr))

    if rmse_lr < mejor_rmse:
        mejor_rmse   = rmse_lr
        mejor_modelo = lr.fit(X, y)
        mejor_set    = (set_name, "LinearReg", cols)

    # -------------------- LightGBM (hyperparameter tuning) -----
    lgb_base = LGBMRegressor(objective="regression",
                             n_estimators=600,
                             n_jobs=-1,
                             random_state=42)

    search_lgb = RandomizedSearchCV(
        estimator=lgb_base,
        param_distributions=param_dist_lgb,
        n_iter=60,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        refit=False,
        verbose=0,
        random_state=42,
    )
    search_lgb.fit(X, y)
    best_lgb_params = search_lgb.best_params_

    best_lgb = lgb_base.set_params(**best_lgb_params)
    rmse_lgb, mape_lgb = evaluar(best_lgb, X, y)
    resultados.append((set_name, "LightGBM", rmse_lgb, mape_lgb))

    if rmse_lgb < mejor_rmse:
        mejor_rmse   = rmse_lgb
        mejor_modelo = best_lgb.fit(X, y)
        mejor_set    = (set_name, "LightGBM", cols)

    # -------------------- MLP (hyperparameter tuning) ----------
    mlp_base = MLPRegressor(max_iter=800, random_state=42)

    search_mlp = RandomizedSearchCV(
        estimator=mlp_base,
        param_distributions=param_dist_mlp,
        n_iter=60,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        refit=False,
        verbose=0,
        random_state=42,
    )
    search_mlp.fit(X, y)
    best_mlp_params = search_mlp.best_params_

    best_mlp = mlp_base.set_params(**best_mlp_params)
    rmse_mlp, mape_mlp = evaluar(best_mlp, X, y)
    resultados.append((set_name, "MLP", rmse_mlp, mape_mlp))

    if rmse_mlp < mejor_rmse:
        mejor_rmse   = rmse_mlp
        mejor_modelo = best_mlp.fit(X, y)
        mejor_set    = (set_name, "MLP", cols)

# =============================================================================
# Step 8: Save artifacts
# =============================================================================
joblib.dump(
    {
        "model"       : mejor_modelo,
        "features"    : mejor_set[2],
        "feature_set" : mejor_set[0],
        "algo"        : mejor_set[1],
    },
    f"meta_global_{mejor_set[1]}.pkl",
)

pd.DataFrame(resultados,
             columns=["FeatureSet", "MetaModel", "RMSE", "MAPE"]
            ).to_csv("resultados_stacking_global.csv", index=False)

print("Meta-modelo global guardado en:",
      f"meta_global_{mejor_set[1]}.pkl")
print("Métricas detalladas → resultados_stacking_global.csv")

