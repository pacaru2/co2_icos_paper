# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Ensemble and Meta-Model Evaluation Pipeline
# Description : 
#   • Loads trained meta-model
#   • Applies hybrid model averaging for available models
#   • Computes ensemble (mean) and meta-model predictions
#   • Saves predictions and computes evaluation metrics
# Author      : Pablo Catret
# =============================================================================

# =============================================================================
# Import libraries
# =============================================================================

import pandas as pd
import numpy as np
import glob, re, joblib, warnings
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# =============================================================================
# Step 1: Load trained meta-model
# =============================================================================
META_PKL = sorted(Path(".").glob("meta_global_*.pkl"))[0]
meta = joblib.load(META_PKL)
meta_model = meta["model"]
meta_features = meta["features"]

# =============================================================================
# Step 2: Locate TEST prediction files (excluding Meta and hybrids)
# =============================================================================
DIR_PRED = Path("EntrenamientoModelos/predicciones")
DIR_METRIC = Path("EntrenamientoModelos/metricas")
DIR_METRIC.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"(.+?)_([^_]+)_TEST\.csv")
csv_files = [
    f for f in glob.glob(str(DIR_PRED / "*_TEST.csv"))
    if "_Meta_" not in Path(f).name and not any(h in Path(f).name for h in ["ProHiTS", "LGBProphet", "ProphetTCN"])
]

stations = sorted({pattern.search(Path(f).name).group(1) for f in csv_files})

# =============================================================================
# Step 3: Utility functions
# =============================================================================
def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2.0))

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred.clip(min=0)) - np.log1p(y_true.clip(min=0))) ** 2))

def add_time_features(df):
    df["month"]     = df.time.dt.month
    # df["quarter"]   = df.time.dt.quarter
    # df["dayofyear"] = df.time.dt.dayofyear
    # df["season"]    = ((df.month % 12) // 3).astype(int)
    # df["sin_month"] = np.sin(2 * np.pi * df.month / 12)
    # df["cos_month"] = np.cos(2 * np.pi * df.month / 12)
    return df

def load_station(st):
    files = [f for f in csv_files if Path(f).name.startswith(st)]
    dfs = []
    for f in files:
        modelo = pattern.search(Path(f).name).group(2)
        df = pd.read_csv(f, parse_dates=["time"])
        dfs.append(df.rename(columns={"pred": f"pred_{modelo}"})[["time", f"pred_{modelo}", "real"]])
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on=["time", "real"], how="inner")
    return base.sort_values("time").reset_index(drop=True)

def save_prediction_and_metrics(df, col_pred, station, model_name):
    df_out = df[["time", col_pred, "real"]].rename(columns={col_pred: "pred"})
    pred_path = DIR_PRED / f"{station}_{model_name}_TEST.csv"
    df_out.to_csv(pred_path, index=False)

    y_true, y_hat = df["real"].values, df_out["pred"].values
    mae  = mean_absolute_error(y_true, y_hat)
    mse  = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_hat) * 100
    smap = smape(y_true, y_hat)
    rls  = rmsle(y_true, y_hat)

    df_metrics = pd.DataFrame([[station, model_name, mae, mse, rmse, mape, smap, rls]],
        columns=["Station", "Model", "MAE", "MSE", "RMSE", "MAPE", "sMAPE", "RMSLE"])
    df_metrics.to_csv(DIR_METRIC / f"{station}_{model_name}_METRICS.csv", index=False)

# =============================================================================
# Step 4: Process each station
# =============================================================================
for st in stations:
    df = load_station(st)
    df = add_time_features(df)

    hybrids = {
        "ProHiTS":    ("pred_Prophet", "pred_NHiTS"),
        "LGBProphet": ("pred_Prophet", "pred_LightGBM"),
        "ProphetTCN": ("pred_Prophet", "pred_TCN"),
    }

    for name, (m1, m2) in hybrids.items():
        if m1 in df.columns and m2 in df.columns:
            df[f"pred_{name}"] = df[[m1, m2]].mean(axis=1)
            save_prediction_and_metrics(df, f"pred_{name}", st, name)
        else:
            print(f"{st}: Cannot create {name} (missing {m1} or {m2})")

    pred_cols = [c for c in df.columns if c.startswith("pred_") and
                 not any(x in c for x in ["Meta", "ProHiTS", "LGBProphet", "ProphetTCN"])]

    if pred_cols:
        df["pred_MediaSimple"] = df[pred_cols].mean(axis=1)
        save_prediction_and_metrics(df, "pred_MediaSimple", st, "MediaSimple")
        print(f"{st}: MediaSimple saved.")
    else:
        print(f"{st}: No pred_ columns found for MediaSimple")

    # Generate prediction with meta-model if features are available
    if all(col in df.columns for col in meta_features):
        X_test = df[meta_features]
        df["pred_Meta"] = meta_model.predict(X_test)
        save_prediction_and_metrics(df, "pred_Meta", st, "Meta")
        print(f"{st}: Meta and hybrids saved.")
    else:
        missing = [c for c in meta_features if c not in df.columns]
        print(f"{st}: Missing columns for meta-model → {missing}")

print("\nProcessing completed for all stations.")
