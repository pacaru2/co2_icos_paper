#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Hyperparameter Tuning via Random Search for 7 Key Stations
# Description : Performs random search over multiple time series models using
#               CO₂ daily data from representative stations.
#               Part of the CO2 Forecast study (Pablo Catret, June 2025).
# Author      : Pablo Catret
# =============================================================================

# =============================================================================
# Import libraries
# =============================================================================
import sys
import os, random, json, logging
from pathlib import Path
from typing import Dict, Any, List
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mse
from darts.models import (
    ExponentialSmoothing, Prophet, LightGBMModel, RandomForest,
    NHiTSModel, TCNModel, TiDEModel, BlockRNNModel
)

# =============================================================================
# Configuration and Constants
# =============================================================================
REPRESENTATIVES = [
    "GAT 216.0", "HPB 131.0", "KRE 50.0", "OXK 23.0",
    "JFJ 13.9", "TRN 50.0", "JUE 120.0",
]
DIR_DATA      = Path("data")
TXT_NUEVOS    = DIR_DATA / "paths_datos_nuevos_2024.txt"
TXT_ANTIGUOS  = DIR_DATA / "paths_datos_antiguos_2024.txt"
LOG_CSV       = Path("resultados/tuning_log.csv")
DIR_MODEL     = Path("modelos_tuning");     DIR_MODEL.mkdir(parents=True, exist_ok=True)
DIR_PRED      = Path("resultados/predicciones"); DIR_PRED.mkdir(exist_ok=True, parents=True)

N_ITER        = 20
SEED          = 42
random.seed(SEED); np.random.seed(SEED)

logging.basicConfig(
    level=logging.DEBUG,
    filename='ajuste_hiper.log',
    format="%(asctime)s – %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)

warnings.filterwarnings("ignore")


def clave_site_altura_to_nombre(txt: Path, base_dir: Path) -> Dict[str, Path]:
    """
    Reads a .txt with paths and returns a dict:
        { "GAT 216.0": Path(...), ... }
    Matches CTS and NRT filenames robustly.
    """
    from pathlib import Path
    import re, logging

    P_CTS = re.compile(r'_([A-Z]{3,4})_(\d+\.\d+)_CTS\.CO2$')
    P_NRT = re.compile(
        r'_([A-Z]{3,4})_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_(\d+\.\d+)_.*\.CO2$'
    )

    def clave(path: Path) -> str | None:
        name = path.name
        m = P_CTS.search(name) or P_NRT.search(name)
        return None if m is None else f"{m.group(1)} {m.group(2)}"

    mapping = {}
    for raw in txt.read_text().splitlines():
        if not raw.strip():
            continue
        p = Path(raw.strip())
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            logging.warning(f"[FICHERO] No existe: {p}")
            continue
        k = clave(p)
        if k:
            mapping[k] = p
    return mapping


PATHS_NEW = clave_site_altura_to_nombre(TXT_NUEVOS,  DIR_DATA)
PATHS_OLD = clave_site_altura_to_nombre(TXT_ANTIGUOS, DIR_DATA)



# =============================================================================
# Utility: Load and preprocess CO₂ daily time series
# =============================================================================
def load_daily_series(path_old: Path, path_new: Path) -> TimeSeries:
    """Reads, cleans and returns a daily CO₂ time series with regular index."""
    def read(p):
        df = pd.read_csv(p, sep=";", comment="#")
        cols = [
            'Site','SamplingHeight','Year','Month','Day','Hour','Minute',
            'DecimalDate','co2'
        ]
        df = df.iloc[:, :len(cols)]
        df.columns = cols[:len(df.columns)]
        df.replace([-999.99, -9.99], np.nan, inplace=True)
        df["TIMESTAMP"] = pd.to_datetime(df[["Year","Month","Day","Hour","Minute"]])
        return df[["TIMESTAMP","co2"]]

    df = pd.concat([read(path_old), read(path_new)])
    df.set_index("TIMESTAMP", inplace=True)
    s = df["co2"].resample("D").mean()

    # Remove long NaN tails (>30 days)
    def find_last_long_nan(series: pd.Series, max_days: int) -> int:
        na_run = 0
        last_end = -1

        for i in range(len(series)):
            if pd.isna(series.iloc[i]):
                na_run += 1
            else:
                if na_run > max_days:
                    last_end = i - 1
                na_run = 0

        if na_run > max_days:
            last_end = len(series) - 1

        return last_end

    max_nan_days = 30
    end_long_nan = find_last_long_nan(s, max_nan_days)

    if end_long_nan != -1:
        logging.debug(f"► Cutting after NaN gap >{max_nan_days} days: positions 0–{end_long_nan}")
        s = s.iloc[end_long_nan + 1:]

    logging.debug(f"  - After cutting: {len(s)} days, NaN={s.isna().sum()}")


    # IQR filter
    q1, q3 = s.quantile([.25, .75]); iqr=q3-q1
    s = s[(s>=q1-1.5*iqr) & (s<=q3+1.5*iqr)]

    # Reindex to daily frequency and interpolate
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)

    s = s.interpolate("pchip")        
    s = s.dropna()

    series = TimeSeries.from_series(s)
    return series


# =============================================================================
# Hyperparameter search space per model
# =============================================================================
SPACE = {
    "ETS":   lambda: {"seasonal_periods": 365},
    "Prophet": lambda: {"seasonality_mode": random.choice(["multiplicative","additive"])},
    "LightGBM": lambda: {
        "lags": 365,
        "output_chunk_length": 365,
        "learning_rate": 10**random.uniform(-2.5,-1),
        "num_leaves": random.randint(15, 63),
        "min_data_in_leaf": random.randint(5, 30),
    },
    "RF": lambda: {
        "lags": 365,
        "output_chunk_length": 365,
        "n_estimators": random.randint(50, 300),
        "max_depth": random.choice([None, 5, 10, 20]),
    },
    "NHiTS": lambda: {
        "input_chunk_length": 366,
        "output_chunk_length": 365,
        "num_blocks": random.randint(1, 3),
        "dropout": random.uniform(0., .3),
    },
    "TCN": lambda: {
        "input_chunk_length": 366,
        "output_chunk_length": 365,
        "num_filters": random.choice([8, 16, 32]),
        "kernel_size": random.choice([2, 3, 4]),
        "dropout": random.uniform(0., .2),
    },
    "TiDE": lambda: {
        "input_chunk_length": 366,
        "output_chunk_length": 365,
        "hidden_size": random.choice([8, 16, 32]),
        "num_encoder_layers": random.randint(1, 2),
    },
    "GRU": lambda: {
        "input_chunk_length": 366,
        "output_chunk_length": 365,
        "hidden_dim": random.randint(1, 64),
        "n_rnn_layers": 1,
    },
    "LSTM": lambda: {
        "input_chunk_length": 366,
        "output_chunk_length": 365,
        "hidden_dim": random.randint(1, 64),
        "n_rnn_layers": 1,
    },
}

MODEL_FACTORY = {
    "ETS": ExponentialSmoothing,
    "Prophet": Prophet,
    "LightGBM": LightGBMModel,
    "RF": RandomForest,
    "NHiTS": NHiTSModel,
    "TCN": TCNModel,
    "TiDE": TiDEModel,
    "GRU": lambda **kw: BlockRNNModel(model="GRU", **kw),
    "LSTM": lambda **kw: BlockRNNModel(model="LSTM", **kw),
}

# =============================================================================
# Random search tuning per station and model
# =============================================================================
def tune_station(name: str, path_old: Path, path_new: Path) -> None:
    logging.info(f"=== {name} ===")

    serie = load_daily_series(path_old, path_new)
    if len(serie) < 1100:
        logging.warning(f"{name}: serie demasiado corta ({len(serie)} d) – se omite")
        return

    scaler = Scaler(); serie_s = scaler.fit_transform(serie)

    train_val, test = serie_s[:-365], serie_s[-365:]
    train, val       = train_val[:-365], train_val[-365:]

    if LOG_CSV.exists():
        tuning_log = pd.read_csv(LOG_CSV)
    else:
        tuning_log = pd.DataFrame(columns=["Station","Model","MSE","Params","PredPath"])

    for model_name in MODEL_FACTORY:
        best_mse   = tuning_log.loc[
            (tuning_log["Station"]==name) & (tuning_log["Model"]==model_name),
            "MSE"
        ].min() if any((tuning_log["Station"]==name) & (tuning_log["Model"]==model_name)) else np.inf
        best_row   = None

        for _ in tqdm(range(N_ITER), desc=f"{name} · {model_name}"):
            params = SPACE[model_name]()
            try:
                model = MODEL_FACTORY[model_name](**params)
                model.fit(train)
                pred  = model.predict(len(val))
                cur_mse = mse(scaler.inverse_transform(val), scaler.inverse_transform(pred))
            except Exception as e:
                logging.debug(f"{model_name} fallo: {e}")
                continue

            if cur_mse < best_mse:
                best_mse = cur_mse
                pred_val  = scaler.inverse_transform(pred).pd_series()
                real_val  = scaler.inverse_transform(val).pd_series()
                df_pred   = pd.DataFrame({"real": real_val, "pred": pred_val})

                mpath = DIR_MODEL / f"{name.replace(' ','_')}_{model_name}.pkl"
                ppath = DIR_PRED  / f"{name.replace(' ','_')}_{model_name}_VAL.csv"
                df_pred.to_csv(ppath)

                best_row = {
                    "Station": name,
                    "Model": model_name,
                    "MSE": best_mse,
                    "Params": json.dumps(params),
                    "PredPath": str(ppath),
                }

        if best_row is not None:
            tuning_log = tuning_log[~((tuning_log["Station"]==name)&(tuning_log["Model"]==model_name))]
            tuning_log = pd.concat([tuning_log, pd.DataFrame([best_row])], ignore_index=True)
            tuning_log.to_csv(LOG_CSV, index=False)
            logging.info(f"{name} · {model_name} – nuevo best MSE={best_mse:.4f}")


# =============================================================================
# Main entry point
# =============================================================================
def main():
    estaciones = sys.argv[1:] if len(sys.argv) > 1 else REPRESENTATIVES

    for est in estaciones:
        est = est.replace("_", " ")
        if est not in PATHS_NEW or est not in PATHS_OLD:
            logging.warning(f"{est}: rutas no encontradas")
            continue
        tune_station(est, PATHS_OLD[est], PATHS_NEW[est])


if __name__ == "__main__":
    main()
