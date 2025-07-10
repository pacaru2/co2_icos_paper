#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Title       : CO₂ Forecasting Model Trainer
# Description : Trains multiple forecasting models on daily CO₂ data for selected stations.
#               Applies tuned hyper-parameters per cluster-representative station.
# Author      : Pablo Catret
# Date        : 2025-07-04
# =============================================================================

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mape, mse, rmse, rmsle, smape
from darts.models import (
    BlockRNNModel,
    ExponentialSmoothing,
    LightGBMModel,
    NHiTSModel,
    Prophet,
    RandomForest,
    TCNModel,
    TiDEModel,
)

# =============================================================================
# Logging & constants
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CLUSTER_TO_REP: Dict[int, str] = {0: "GAT", 1: "HPB", 2: "KRE", 3: "OXK", 4: "JFJ", 5: "TRN", 6: "JUE"}
SUPPORTED_MODELS: List[str] = [
    "TiDE",
    "GRU",
    "Prophet",
    "ETS",
    "LightGBM",
    "RF",
    "NHiTS",
    "TCN",
    "LSTM",
]

BASE_DIR = Path("EntrenamientoModelos")
METRICS_DIR = BASE_DIR / "metricas"
PRED_DIR = BASE_DIR / "predicciones"
MODEL_DIR = BASE_DIR / "modelos"

for d in (METRICS_DIR, PRED_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utility functions
# =============================================================================

def safe_station_name(name: str) -> str:
    """Return a filename‑safe version of the station name («BIR 50.0» → «BIR_50.0»)."""
    return name.replace(" ", "_")


def load_cluster_mapping(path: str | Path = "resumen_estaciones.csv") -> Dict[str, int]:
    """Return a mapping {NombreCompleto: cluster_number}."""
    df = pd.read_csv(path, usecols=["NombreCompleto", "Cluster"])
    return dict(zip(df["NombreCompleto"], df["Cluster"].astype(int)))


def representative_for_station(station: str, clusters: Dict[str, int]) -> str:
    """Get the representative code (e.g. "GAT") for *station* using the cluster table."""
    try:
        cluster = clusters[station]
    except KeyError as exc:
        raise KeyError(f"Station '{station}' not found in resumen_estaciones.csv") from exc

    try:
        return CLUSTER_TO_REP[cluster]
    except KeyError as exc:
        raise KeyError(f"Cluster id '{cluster}' has no representative mapping") from exc


def load_best_params(rep_code: str) -> Dict[str, Dict]:
    """Load best hyper‑parameters for every model from the representative tuning log."""
    tl_path = Path("SeleccionHiperparametros") / "Hiperparametros" / rep_code / "tuning_log.csv"
    if not tl_path.exists():
        raise FileNotFoundError(f"Tuning log not found: {tl_path}")

    df = pd.read_csv(tl_path, usecols=["Model", "Params"])
    best: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        model_name: str = row["Model"].strip()
        try:
            params_dict: Dict = ast.literal_eval(row["Params"])
        except Exception as exc:
            logging.warning("Unable to parse params for %s: %s", model_name, exc)
            params_dict = {}
        best[model_name] = params_dict

    return best


def build_model(model_name: str, params: Dict) -> "ForecastingModel":
    """Instantiate a Darts model given its *model_name* and tuned *params*."""

    common = {
        k: params.pop(k) for k in list(params.keys()) if k in {"input_chunk_length", "output_chunk_length", "lags"}
    }

    if model_name == "ETS":
        return ExponentialSmoothing(**params)
    if model_name == "Prophet":
        return Prophet(**params)
    if model_name == "LightGBM":
        return LightGBMModel(**common, **params)
    if model_name == "RF":
        return RandomForest(**common, **params)
    if model_name == "NHiTS":
        return NHiTSModel(**common, **params)
    if model_name == "TCN":
        return TCNModel(**common, **params)
    if model_name == "TiDE":
        return TiDEModel(**common, **params)
    if model_name == "GRU":
        return BlockRNNModel(model="GRU", **common, **params)
    if model_name == "LSTM":
        return BlockRNNModel(model="LSTM", **common, **params)

    raise ValueError(f"Unsupported model: {model_name}")


# =============================================================================
# Data preparation helpers (identical logic to the original script)
# =============================================================================

def read_raw_station_data(old_path: str, new_path: str) -> pd.DataFrame:
    """Read, concatenate and return the raw hourly data for a station."""

    def load(path: str) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            header_idx = next(i for i, line in enumerate(fh) if not line.startswith("#"))
        df_tmp = pd.read_csv(path, sep=";", comment="#")
        columns = [
            "Site",
            "SamplingHeight",
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "DecimalDate",
            "co2",
            "Stdev",
            "NbPoints",
            "Flag",
            "InstrumentId",
            "QualityId",
            "LTR",
            "CMR",
            "STTB",
            "QcBias",
            "QcBiasUncertainty",
            "co2-WithoutSpikes",
            "Stdev-WithoutSpikes",
            "NbPoints-WithoutSpikes",
        ][: len(df_tmp.columns)]
        df_tmp.columns = columns
        return df_tmp

    df_old = load("data/"+old_path)
    df_new = load("data/"+new_path)
    df = pd.concat([df_old, df_new], ignore_index=True)

    df.replace({-999.99: np.nan, -9.99: np.nan}, inplace=True)

    df["TIMESTAMP"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df.drop(["Year", "Month", "Day", "Hour", "Minute"], axis=1, inplace=True)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned daily dataframe with columns TIMESTAMP + valor."""
    df_daily = df.set_index("TIMESTAMP")[["co2"]].resample("D").mean()

    def find_last_long_nan_interval(series: pd.Series, max_days: int = 60) -> int:
        na_run, last_end = 0, -1
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

    idx_cut = find_last_long_nan_interval(df_daily["co2"])
    if idx_cut != -1:
        df_daily = df_daily.iloc[idx_cut + 1 :]

    q1, q3 = df_daily["co2"].quantile([0.25, 0.75])
    iqr = q3 - q1
    bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    mask = df_daily["co2"].between(bounds[0], bounds[1])
    df_daily.loc[~mask, "co2"] = np.nan

    df_daily["co2_interpolated"] = df_daily["co2"].interpolate(method="pchip")

    tidy = (
        df_daily.reset_index()
        .dropna(subset=["co2_interpolated"])
        .rename(columns={"co2_interpolated": "valor"})[["TIMESTAMP", "valor"]]
    )

    return tidy


def train_and_evaluate(
    station: str,
    df: pd.DataFrame,
    best_params: Dict[str, Dict],
    metrics_dir: Path = METRICS_DIR,
    pred_dir: Path = PRED_DIR,
    model_dir: Path = MODEL_DIR,
):
    """Train every supported model on df and persist metrics + test predictions."""

    logging.info("Training models for %s", station)

    series = TimeSeries.from_dataframe(df, time_col="TIMESTAMP", value_cols="valor")
    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)

    train, test = series_scaled.split_before(len(series_scaled) - 365)

    metrics_records = []
    for model_name in SUPPORTED_MODELS:
        params = best_params.get(model_name, {})
        try:
            model = build_model(model_name, params.copy())
        except Exception as exc:
            logging.warning("Skipping %s for %s – cannot build model (%s)", model_name, station, exc)
            continue

        try:
            model.fit(train)
            forecast = model.predict(len(test))
        except Exception as exc:
            logging.warning("Training failed for %s – skipping (%s)", model_name, exc)
            continue

        forecast = scaler.inverse_transform(forecast)
        test_unscaled = scaler.inverse_transform(test)

        res = {
            "Station": station,
            "Model": model_name,
            "MAE": mae(test_unscaled, forecast),
            "MSE": mse(test_unscaled, forecast),
            "RMSE": rmse(test_unscaled, forecast),
            "MAPE": mape(test_unscaled, forecast),
            "sMAPE": smape(test_unscaled, forecast),
            "RMSLE": rmsle(test_unscaled, forecast),
        }
        metrics_records.append(res)

        st_safe = safe_station_name(station)
        metrics_path = metrics_dir / f"{st_safe}_{model_name}_METRICS.csv"
        pd.DataFrame([res]).to_csv(metrics_path, index=False)

        pred_path = pred_dir / f"{st_safe}_{model_name}_TEST.csv"
        pd.DataFrame(
            {
                "time": test_unscaled.time_index,
                "real": test_unscaled.values().flatten(),
                "pred": forecast.values().flatten(),
            }
        ).to_csv(pred_path, index=False)

        model_path = model_dir / f"{st_safe}_{model_name}.pkl"
        joblib.dump(model, model_path)

        logging.info("Saved metrics, predictions and model for %s – %s", station, model_name)

    if metrics_records:
        station_metrics_all = pd.DataFrame(metrics_records)
        agg_path = metrics_dir / "ALL_METRICS.csv"
        if agg_path.exists():
            station_metrics_all.to_csv(agg_path, mode="a", header=False, index=False)
        else:
            station_metrics_all.to_csv(agg_path, index=False)


# =============================================================================
# Main entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tuned models for selected stations.")
    p.add_argument(
        "stations",
        nargs="*",
        help="Station names with underscore or space as separator, e.g. 'BIR_50.0' or 'BIR 50.0'.",
    )
    return p.parse_args()


allowed_names: List[str] = [
    "BIR 10.0",
    "BIR 50.0",
    "BIR 75.0",
    "GAT 132.0",
    "GAT 216.0",
    "GAT 30.0",
    "GAT 341.0",
    "GAT 60.0",
    "HPB 131.0",
    "HPB 50.0",
    "HPB 93.0",
    "HTM 150.0",
    "HTM 70.0",
    "IPR 100.0",
    "IPR 40.0",
    "IPR 60.0",
    "JFJ 13.9",
    "JUE 120.0",
    "JUE 50.0",
    "JUE 80.0",
    "KIT 100.0",
    "KIT 200.0",
    "KIT 60.0",
    "KRE 10.0",
    "KRE 125.0",
    "KRE 250.0",
    "KRE 50.0",
    "LMP 8.0",
    "LUT 60.0",
    "NOR 100.0",
    "NOR 32.0",
    "NOR 58.0",
    "OPE 10.0",
    "OPE 120.0",
    "OPE 50.0",
    "OXK 163.0",
    "OXK 23.0",
    "OXK 90.0",
    "PAL 12.0",
    "PUI 47.0",
    "PUI 84.0",
    "PUY 10.0",
    "RGL 45.0",
    "RGL 90.0",
    "SAC 100.0",
    "SAC 15.0",
    "SAC 60.0",
    "SMR 125.0",
    "SMR 16.8",
    "SMR 67.2",
    "SNO 20.0",
    "SNO 50.0",
    "SNO 85.0",
    "SSL 12.0",
    "SSL 35.0",
    "STE 127.0",
    "STE 187.0",
    "STE 252.0",
    "STE 32.0",
    "TOH 10.0",
    "TOH 110.0",
    "TOH 147.0",
    "TOH 76.0",
    "TRN 100.0",
    "TRN 180.0",
    "TRN 5.0",
    "TRN 50.0",
    "UTO 57.0",
    "WES 14.0",
    "ZSF 3.0",
]


def main() -> None:
    args = parse_args()
    requested_stations = [s.replace("_", " ") for s in args.stations] if args.stations else allowed_names

    cluster_map = load_cluster_mapping()
    with open("data/estaciones_alturas.txt", "r", encoding="utf-8") as f:
        all_names = [x.strip() for x in f]
    with open("data/paths_datos_antiguos_2024.txt", "r", encoding="utf-8") as f:
        old_paths = [x.strip() for x in f]
    with open("data/paths_datos_nuevos_2024.txt", "r", encoding="utf-8") as f:
        new_paths = [x.strip() for x in f]

    name_to_paths = dict(zip(all_names, zip(old_paths, new_paths)))

    for station in requested_stations:
        if station not in name_to_paths:
            logging.warning("Station %s not present in data path lists – skipping", station)
            continue

        old_path, new_path = name_to_paths[station]

        try:
            rep = representative_for_station(station, cluster_map)
            best_params = load_best_params(rep)
        except Exception as exc:
            logging.error("Failed to prepare tuning data for %s: %s", station, exc)
            continue

        try:
            raw_df = read_raw_station_data(old_path, new_path)
            tidy_df = preprocess(raw_df)
        except Exception as exc:
            logging.error("Data preparation failed for %s: %s", station, exc)
            continue

        try:
            train_and_evaluate(station, tidy_df, best_params)
        except Exception as exc:
            logging.error("Training/evaluation failed for %s: %s", station, exc)
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
