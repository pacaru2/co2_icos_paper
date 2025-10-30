# =============================================================================
# Title       : MaxAE Calculation
# Description : Calculates the MaxAE for each model and series.
# Author      : Pablo Catret
# =============================================================================

import os
import re
import pandas as pd
import numpy as np

# Directorios base
PRED_DIR = "EntrenamientoModelos/predicciones"
METRICS_DIR = "EntrenamientoModelos/metricas"
OUTPUT_DIR = os.path.join(METRICS_DIR, "max_ae")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")

# Crear carpetas de salida si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ------- LISTA BLANCA DE ESTACIONES (abreviaturas ICOS) -------
STATION_WHITELIST = {
    "BIR","GAT","HPB","HTM","IPR","JFJ","JUE","KIT","KRE","LMP",
    "LUT","NOR","OPE","OXK","PAL","PUI","PUY","RGL","SAC","SMR",
    "SNO","SSL","STE","TOH","TRN","UTO","WES","ZSF"
}
# --------------------------------------------------------------


def find_columns(df: pd.DataFrame):
    """Detecta automáticamente columnas de reales/predichos."""
    cols_lower = [c.lower() for c in df.columns]
    y_true_col, y_pred_col = None, None
    true_keys = ("real", "y_true", "true", "observ", "target")
    pred_keys = ("pred", "y_pred", "forecast", "estim", "prediction")
    for i, c in enumerate(cols_lower):
        if any(k in c for k in true_keys) and y_true_col is None:
            y_true_col = df.columns[i]
        if any(k in c for k in pred_keys) and y_pred_col is None:
            y_pred_col = df.columns[i]
    return y_true_col, y_pred_col


def compute_max_ae(pred_file: str):
    """Calcula el MaxAE de un archivo de predicciones."""
    try:
        df = pd.read_csv(pred_file)
    except Exception as e:
        print(f"[WARNING] No se pudo leer {pred_file}: {e}")
        return None
    y_true_col, y_pred_col = find_columns(df)
    if not y_true_col or not y_pred_col:
        print(f"[WARNING] No se detectaron columnas válidas en {pred_file}")
        return None
    df = df.dropna(subset=[y_true_col, y_pred_col])
    if df.empty:
        print(f"[WARNING] Sin datos válidos tras dropna() en {pred_file}")
        return None
    max_ae = np.abs(
        df[y_true_col].to_numpy(dtype=float) - df[y_pred_col].to_numpy(dtype=float)
    ).max()
    return float(max_ae)


def parse_station_model_from_filename(name: str):
    """Extrae (station, model) del patrón {station}_{model}_METRICS.csv o *_TEST.csv"""
    base = os.path.basename(name)
    base = re.sub(r"(_METRICS|_TEST)\.csv$", "", base, flags=re.IGNORECASE)
    parts = base.split("_")
    if len(parts) >= 2:
        station, model = parts[0], parts[-1]
        return station, model
    return None, None


def parse_model_from_metrics_df_or_name(df: pd.DataFrame, fname: str) -> str:
    """Prefiere columna 'model' si existe; si no, la extrae del nombre."""
    if "model" in df.columns and df["model"].nunique() == 1:
        return str(df["model"].iloc[0])
    _, model = parse_station_model_from_filename(fname)
    return model or "UNKNOWN"


def summarize_maxae_from_output():
    """
    Lee todos los *_METRICS.csv en OUTPUT_DIR (con columna MaxAE),
    filtra SOLO las estaciones en STATION_WHITELIST y genera
    medianas, Q1, Q3 de MaxAE por modelo.
    """
    rows = []
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith("_METRICS.csv"):
            continue

        station, _ = parse_station_model_from_filename(fname)
        if station is None or station not in STATION_WHITELIST:
            continue

        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARNING] No se pudo leer {fpath}: {e}")
            continue

        if "MaxAE" not in df.columns:
            print(f"[INFO] Se ignora {fpath} (no tiene columna MaxAE).")
            continue

        model_name = parse_model_from_metrics_df_or_name(df, fname)
        vals = pd.to_numeric(df["MaxAE"], errors="coerce").dropna()
        if vals.empty:
            continue

        for v in vals:
            rows.append({"station": station, "model": model_name, "MaxAE": float(v)})

    if not rows:
        print("[INFO] No hay datos para resumen (tras filtrar por estaciones).")
        return

    df_all = pd.DataFrame(rows)
    summary = (
        df_all.groupby("model", as_index=False)
        .agg(
            median_MaxAE=("MaxAE", "median"),
            q25_MaxAE=("MaxAE", lambda x: np.quantile(x, 0.25)),
            q75_MaxAE=("MaxAE", lambda x: np.quantile(x, 0.75)),
        )
        .sort_values(["median_MaxAE", "model"])
        .reset_index(drop=True)
    )

    out_summary = os.path.join(SUMMARY_DIR, "median_maxae_by_model.csv")
    summary.to_csv(out_summary, index=False)
    print(f"[OK] Resumen (mediana/Q1/Q3 de MaxAE por modelo) guardado en: {out_summary}")


def concatenate_all_metrics():
    """
    Concatena TODOS los CSV de métricas ya generados dentro de OUTPUT_DIR
    (normalmente {station}_{model}_METRICS.csv) en un único archivo:
    OUTPUT_DIR/all_metrics_maxae.csv
    """
    csv_paths = []
    for fname in os.listdir(OUTPUT_DIR):
        # Incluimos solo CSV de métricas individuales (excluye resúmenes y el propio all_*)
        if fname.endswith("_METRICS.csv") and not fname.startswith("all_"):
            csv_paths.append(os.path.join(OUTPUT_DIR, fname))

    if not csv_paths:
        print("[INFO] No se encontraron CSVs de métricas para concatenar en OUTPUT_DIR.")
        return

    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            # Añade contexto útil por si faltara en alguna métrica
            station, model = parse_station_model_from_filename(os.path.basename(path))
            if "station" not in df.columns and station is not None:
                df["station"] = station
            if "model" not in df.columns and model is not None:
                df["model"] = model
            frames.append(df)
        except Exception as e:
            print(f"[WARNING] No se pudo leer {path}: {e}")

    if not frames:
        print("[INFO] No se pudo leer ningún CSV de métricas válido para concatenar.")
        return

    # Concatenación directa (columnas = unión)
    df_all = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    out_path = os.path.join(OUTPUT_DIR, "all_metrics_maxae.csv")
    try:
        df_all.to_csv(out_path, index=False)
        print(f"[OK] Concatenación guardada en: {out_path} (n_filas={len(df_all)})")
    except Exception as e:
        print(f"[WARNING] No se pudo escribir {out_path}: {e}")


def process_all():
    """
    Calcula MaxAE para cada *_TEST.csv, actualiza métricas y guarda
    en EntrenamientoModelos/metricas/max_ae/, luego genera resumen
    y la concatenación de todas las métricas (all_metrics_maxae.csv).
    """
    any_processed = False
    for pred_file in os.listdir(PRED_DIR):
        if not pred_file.endswith("_TEST.csv"):
            continue

        base_name = pred_file.replace("_TEST.csv", "")
        pred_path = os.path.join(PRED_DIR, pred_file)
        metrics_name = f"{base_name}_METRICS.csv"
        metrics_path = os.path.join(METRICS_DIR, metrics_name)

        if not os.path.exists(metrics_path):
            print(f"[INFO] No se encontró el CSV de métricas para {base_name}")
            continue

        max_ae = compute_max_ae(pred_path)
        if max_ae is None:
            continue

        try:
            df_metrics = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"[WARNING] No se pudo leer métricas {metrics_path}: {e}")
            continue

        df_metrics["MaxAE"] = max_ae
        out_path = os.path.join(OUTPUT_DIR, metrics_name)
        try:
            df_metrics.to_csv(out_path, index=False)
        except Exception as e:
            print(f"[WARNING] No se pudo escribir {out_path}: {e}")
            continue

        any_processed = True
        print(f"[OK] Guardado {out_path} con MaxAE={max_ae:.6f}")

    # Genera siempre la concatenación, haya o no nuevos archivos procesados
    concatenate_all_metrics()

    if any_processed:
        summarize_maxae_from_output()
    else:
        print("[INFO] No se procesó ningún *_TEST.csv; se generó (si procede) la concatenación, pero no el resumen.")


if __name__ == "__main__":
    process_all()
