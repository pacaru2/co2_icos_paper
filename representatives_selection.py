#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Station Clustering & Representative Selector
# Description : Selects representative stations for CO₂ forecasting by applying
#               mixed clustering (categorical + numerical) and data validation.
# Author      : Pablo Catret
# =============================================================================

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import re
import logging
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import matplotlib.pyplot as plt

# ==============================================================================
# BASIC CONFIGURATION
# ==============================================================================

logging.basicConfig(
    level=logging.DEBUG,
    filename='mi_log.log',
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# INPUT / OUTPUT PATHS
# ==============================================================================

DIR_DATA = Path("data")
PATH_CLASIF = DIR_DATA / "ClasificacionesEstacion.csv"
PATH_NUEVOS = DIR_DATA / "paths_datos_nuevos_2024.txt"
PATH_ANTIGUOS = DIR_DATA / "paths_datos_antiguos_2024.txt"
PATH_NOMBRES = DIR_DATA / "estaciones_alturas.txt"

DIR_RESULTADOS = Path("resultados")
DIR_RESULTADOS.mkdir(exist_ok=True, parents=True)
PATH_REPRESENTANTES = DIR_RESULTADOS / "representantes_por_cluster.csv"
PATH_RESUMEN = DIR_RESULTADOS / "resumen_estaciones.csv"

# ==============================================================================
# 1. BUILD FEATURE TABLE
# ==============================================================================

def parse_nombre(nombre: str) -> Tuple[str, float]:
    """
    Parses station code (3–4 letters) and sampling height from string.
    Example: "BIR 10.0" → ("BIR", 10.0)
    """
    match = re.match(r"([A-Z]{3,4})\s+([0-9]+(?:\.[0-9]+)?)", nombre.strip())
    if not match:
        raise ValueError(f"No se pudo parsear el nombre {nombre}")
    codigo, altura = match.group(1), float(match.group(2))
    return codigo, altura


def construir_features() -> pd.DataFrame:
    """
    Returns a DataFrame with the selected 4 features for clustering.
    Combines data from names list and classification table.
    """
    nombres = [ln.strip() for ln in PATH_NOMBRES.read_text().splitlines() if ln.strip()]

    clasif = pd.read_csv(PATH_CLASIF, sep=";", decimal=",")
    clasif.rename(columns={"Abbreviation": "Codigo"}, inplace=True)

    registros = []
    for nom in nombres:
        codigo, altura_muestreo = parse_nombre(nom)
        fila = clasif.loc[clasif["Codigo"] == codigo].copy()
        if fila.empty:
            logging.warning(f"{codigo} no está en ClasificacionesEstacion.csv – se ignora")
            continue
        fila = fila.iloc[0].to_dict()
        fila.update({
            "NombreCompleto": nom,
            "SamplingHeight": altura_muestreo,
        })
        registros.append(fila)

    df = pd.DataFrame(registros)
    df = df[[
        "NombreCompleto", "Codigo",
        "ESA", "Koppen",
        "Level", "SamplingHeight"
    ]]
    return df


# ==============================================================================
# 2. MIXED CLUSTERING (NUMERIC + CATEGORICAL)
# ==============================================================================

def crear_matriz_kprototypes(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Tuple[np.ndarray, List[int]]:
    """
    Builds input matrix for KPrototypes clustering:
    - Categorical variables as strings
    - Numerical variables are scaled
    Returns the combined matrix and positions of categorical columns.
    """
    df_cat = df[cat_cols].astype(str)
    df_num = StandardScaler().fit_transform(df[num_cols])
    X = np.concatenate([df_cat.values, df_num], axis=1)
    cat_idx = list(range(len(cat_cols)))  # las categóricas van primero
    return X, cat_idx


def clusterizar(df_feat: pd.DataFrame, k_min=2, k_max=12, alpha = 5) -> Tuple[pd.DataFrame, KPrototypes]:
    """
    Performs KPrototypes clustering for optimal k using cost + penalty.
    Adds 'Cluster' column to the feature DataFrame.
    """
    cat_cols = ["ESA", "Koppen"]
    num_cols = ["Level", "SamplingHeight"]

    X, cat_idx = crear_matriz_kprototypes(df_feat, cat_cols, num_cols)

    mejor_k, mejor_coste, mejor_modelo = None, np.inf, None
    for k in range(k_min, min(k_max, len(df_feat)) + 1):
        modelo = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=42)
        clusters = modelo.fit_predict(X, categorical=cat_idx)
        coste = modelo.cost_ + alpha * k
        logging.info(f"k={k} → coste={coste:.2f}")
        if coste < mejor_coste:
            mejor_k, mejor_coste, mejor_modelo = k, coste, modelo

    logging.info(f"Mejor k={mejor_k} con coste={mejor_coste:.2f}")
    df_feat["Cluster"] = mejor_modelo.predict(X, categorical=cat_idx)
    return df_feat, mejor_modelo

def plot_clusters_pca(df_feat: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> None:
    """
    Projects stations into 2D space using PCA on scaled numeric features.
    Plots clusters with different colors and ESA shapes.
    """
    X_scaled = StandardScaler().fit_transform(df_feat[num_cols])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = df_feat.copy()
    df_plot["PCA1"] = X_pca[:, 0]
    df_plot["PCA2"] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', 'P', 'h', '+']
    esas = sorted(df_plot["ESA"].unique())
    clusters = sorted(df_plot["Cluster"].unique())

    for i, esa in enumerate(esas):
        for cluster in clusters:
            subset = df_plot[(df_plot["ESA"] == esa) & (df_plot["Cluster"] == cluster)]
            ax.scatter(
                subset["PCA1"], subset["PCA2"],
                label=f"{esa} - C{cluster}",
                marker=markers[i % len(markers)],
                alpha=0.7
            )

    ax.set_title("Clusters de estaciones (PCA)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=5)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("PCA.png")

# ==============================================================================
# 3. COUNT VALID DAYS PER STATION-HEIGHT
# ==============================================================================

def cargar_df_csv(path: Path) -> pd.DataFrame:
    """
    Loads an ICOS-format CSV file and renames standard columns.
    Expected output columns:
        Site, SamplingHeight, Year, Month, Day, Hour, Minute, co2
    """
    df = pd.read_csv(path, sep=";", comment="#")
    columnas_base = [
        'Site', 'SamplingHeight', 'Year', 'Month', 'Day', 'Hour', 'Minute',
        'DecimalDate', 'co2', 'Stdev', 'NbPoints', 'Flag', 'InstrumentId',
        'QualityId', 'LTR', 'CMR', 'STTB', 'QcBias', 'QcBiasUncertainty',
        'co2-WithoutSpikes', 'Stdev-WithoutSpikes', 'NbPoints-WithoutSpikes'
    ]
    df.columns = columnas_base[:len(df.columns)]
    df = df.replace([-999.99, -9.99], np.nan)
    return df


def dias_validos_postpro(path_antiguo: Path, path_nuevo: Path, max_nan_days: int = 30) -> int:
    """
    Reconstructs the postprocessing pipeline and returns number of valid days.
    Logs detailed steps including cleaning, resampling, and filtering.
    """
    logger.debug(f"► Cargando archivos: {path_antiguo.name}, {path_nuevo.name}")
    df_a, df_n = cargar_df_csv(path_antiguo), cargar_df_csv(path_nuevo)
    logger.debug(f"  - Antiguo shape: {df_a.shape} | Nuevo shape: {df_n.shape}")

    df = pd.concat([df_a, df_n], ignore_index=True)
    logger.debug(f"► Concatenado shape: {df.shape}")

    df["TIMESTAMP"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df.drop(["Year", "Month", "Day", "Hour", "Minute"], axis=1, inplace=True)
    df.set_index("TIMESTAMP", inplace=True)
    logger.debug(f"► DataFrame indexado; rango {df.index.min()} → {df.index.max()}")

    df_d = df[["co2"]].resample("D").mean()
    logger.debug(f"► Resample diario: {df_d.shape[0]} días, NaN={df_d['co2'].isna().sum()}")

    def find_last_long_nan(series, max_days):
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


    end_long_nan = find_last_long_nan(df_d["co2"], max_days=max_nan_days)

    if end_long_nan != -1:
        logger.debug(f"► Corte tras hueco largo >{max_nan_days} d: posiciones 0–{end_long_nan}")
        df_d = df_d.iloc[end_long_nan + 1:]

    logger.debug(f"  - Tras corte: {df_d.shape[0]} días, NaN={df_d['co2'].isna().sum()}")

    Q1, Q3 = df_d["co2"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    logger.debug(f"► IQR filter: Q1={Q1:.2f}, Q3={Q3:.2f}, bounds=({lower:.2f}, {upper:.2f})")
    before = len(df_d)
    df_d = df_d[(df_d["co2"] >= lower) & (df_d["co2"] <= upper)]
    outliers_eliminados = before - len(df_d)

    logger.debug(f"  - Filtrados {outliers_eliminados} outliers")

    nan_before = df_d["co2"].isna().sum()
    df_d["co2"] = df_d["co2"].interpolate(method="pchip")
    df_d.dropna(inplace=True)
    logger.debug(f"► Interpolación PCHIP: NaN antes={nan_before}, después={df_d['co2'].isna().sum()}")

    dias_validos = len(df_d)
    logger.info(f"✔ {path_antiguo.stem.split('_')[0]} – días válidos: {dias_validos}")
    
    return dias_validos, outliers_eliminados


logger = logging.getLogger(__name__)

# ==============================================================================
# 4. HELPER FUNCTIONS FOR FILE MAPPING
# ==============================================================================

P_CTS = re.compile(r'_([A-Z]{3,4})_(\d+\.\d+)_CTS\.CO2$')

P_NRT = re.compile(
    r'_([A-Z]{3,4})_'           
    r'\d{4}-\d{2}-\d{2}_'        
    r'\d{4}-\d{2}-\d{2}_'        
    r'(\d+\.\d+)_.*\.CO2$'      
)

def clave_site_altura(path: Path) -> str | None:
    """
    Extracts site and height key from filename (e.g., 'BIR_10.0').
    Returns None if the pattern is not recognized.
    """
    name = path.name
    m = P_CTS.search(name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"         

    m = P_NRT.search(name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    logger.warning(f"[CLAVE] Nombre de archivo no reconocido: '{name}'")
    return None



# ---------------------------------------------------------------------------
# 2. Construir diccionarios {clave: Path}
# ---------------------------------------------------------------------------

def build_paths_dict(txt_path: Path, base_dir: Path) -> dict[str, Path]:
    """
    Reads .txt list and returns dictionary {key: Path}.
    Prepends base_dir to relative paths. Ignores missing files.
    """
    mapping = {}
    for raw in txt_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue

        p = Path(raw)
        if not p.is_absolute():
            p = base_dir / p

        if not p.exists():
            logger.warning(f"[FICHERO] No encontrado: {p}")
            continue

        key = clave_site_altura(p)
        if not key:
            continue
        if key in mapping:
            logger.warning(f"Clave duplicada {key} — se mantiene el primer path")
            continue
        mapping[key] = p
    return mapping





# ==============================================================================
# 5. MAIN FUNCTION TO ADD VALID DAYS TO FEATURES
# ==============================================================================

def conteo_dias(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'DiasValidos' and 'OutliersEliminados' columns to the features DataFrame.
    Uses old and new CO2 file paths to count valid measurement days.
    """
    nuevos_paths   = build_paths_dict(PATH_NUEVOS,   base_dir=DIR_DATA)
    antiguos_paths = build_paths_dict(PATH_ANTIGUOS, base_dir=DIR_DATA)



    logger.debug(f"{len(nuevos_paths)} ficheros nuevos mapeados")
    logger.debug(f"{len(antiguos_paths)} ficheros antiguos mapeados")

    dias_validos_list: list[int] = []
    outliers_list: list[int] = []

    for row in df_feat.itertuples(index=False):
        nombre = row.NombreCompleto          
        site, altura = nombre.split()        
        key = f"{site}_{altura}"             

        ant_path = antiguos_paths.get(key)
        nue_path = nuevos_paths.get(key)

        if ant_path is None or nue_path is None:
            logger.warning(
                f"Faltan ficheros para {nombre}: "
                f"{'antiguo NO' if ant_path is None else ''} "
                f"{'nuevo NO'   if nue_path  is None else ''}"
            )
            dias_validos_list.append(0)
            continue

        try:
            dias, outliers = dias_validos_postpro(ant_path, nue_path)
        except Exception as e:
            logger.error(
                f"Error procesando {nombre} "
                f"(antiguo='{ant_path.name}', nuevo='{nue_path.name}'): {e}",
                exc_info=True
            )
            dias, outliers = 0, 0

        dias_validos_list.append(dias)
        outliers_list.append(outliers)

    df_feat["DiasValidos"] = dias_validos_list
    df_feat["OutliersEliminados"] = outliers_list

    return df_feat



# ==============================================================================
# 6. SELECT REPRESENTATIVE STATIONS PER CLUSTER
# ==============================================================================

def elegir_representantes(df_feat: pd.DataFrame, minimo_dias=1460) -> pd.DataFrame:
    """
    Selects one representative station per cluster based on:
    1. Having enough valid days (if available)
    2. Not being 'IPR'
    3. Minimum Euclidean distance to numeric centroid (scaled)
    """
    num_cols = ["Level", "SamplingHeight"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat[num_cols])
    df_feat_scaled = df_feat.copy()
    df_feat_scaled[num_cols] = X_scaled

    reps = []
    for cl, subset in df_feat_scaled.groupby("Cluster"):
        subset = subset.copy()
        candidatas = subset[subset["DiasValidos"] >= minimo_dias]
        candidatas = candidatas[~candidatas["NombreCompleto"].str.contains("IPR", na=False)]
        
        if candidatas.empty:
            candidatas = subset[~subset["NombreCompleto"].str.contains("IPR", na=False)]
            if candidatas.empty:
                candidatas = subset

        centroide = candidatas[num_cols].mean().values.reshape(1, -1)
        distancias = cdist(candidatas[num_cols], centroide)
        idx_min = distancias.argmin()
        ganador = candidatas.iloc[idx_min]
        reps.append(ganador)

    df_reps = pd.DataFrame(reps).reset_index(drop=True)
    return df_reps



# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

def main():
    logging.info("Construyendo tabla de características…")
    df_feat = construir_features()

    logging.info("Realizando clustering…")
    df_feat, _ = clusterizar(df_feat)

    logging.info("Contando días válidos por estación…")
    df_feat = conteo_dias(df_feat)

    logging.info("Eligiendo representantes por cluster…")
    df_reps = elegir_representantes(df_feat)

    logging.info("Graficando PCA…")
    plot_clusters_pca(df_feat, cat_cols=["ESA", "Koppen"], num_cols=["Level", "SamplingHeight"])

    df_reps.to_csv(PATH_REPRESENTANTES, index=False)

    logging.info(f"Representantes guardados en {PATH_REPRESENTANTES}")
    df_feat[["NombreCompleto", "DiasValidos", "OutliersEliminados", "Cluster"]].to_csv(
        PATH_RESUMEN, index=False
    )
    logging.info(f"Clusters guardados en {PATH_RESUMEN}")

    print("\n=== REPRESENTANTES SELECCIONADOS ===")
    print(df_reps[[
        "Cluster", "NombreCompleto", "ESA", "Koppen",
        "Level", "SamplingHeight", "DiasValidos"
    ]].to_markdown(index=False))


if __name__ == "__main__":
    main()
